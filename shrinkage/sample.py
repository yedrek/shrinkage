import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from sklearn.isotonic import IsotonicRegression


class SampleEigenvalues:
    """
    Instantiate this class with a data matrix Y of shape N x T_total.
    Based on it:
      - calculate the sample estimator E and its eigensystem
        for the final N x T part of Y;
      - optionally (by providing T_out = length of each out-of-sample fold),
        estimate the oracle eigenvalues,
        via the moving-window K-fold cross-validation procedure;
      - then also fit isotonic regression to such oracle eigenvalues.
    Furthermore, estimate the eigenvalue density and Hilbert transform of these eigenvalues
    via the (adaptive) Epanechnikov kernel method
    [Ledoit, O., Wolf, M., Analytical nonlinear shrinkage of large-dimensional covariance matrices].
    Optionally, remove n_top of the top eigenvalues from any calculation performed here;
    this removes potential outliers from the analysis.
    """
    def __init__(
        self,
        Y,
        T,
        T_out=None,
        n_top=None,
        name='sample'
        ):

        self.name = name
        
        self.Y = Y
        self.N, self.T_total = self.Y.shape
        self.T = T
        self.q = self.N / self.T
        self.T_out = T_out
        self.n_top = n_top

        assert self.T <= self.T_total

        self.calculate_E_final()

        # if T_out is given, it means we want to perform
        # the moving-window K-fold cross-validation estimation
        # of the oracle eigenvalues

        if self.T_out is not None:
            assert self.T_out <= self.T_total - self.T
            self.calculate_oracle_mwcv()
            self.calculate_oracle_mwcv_iso()
        
        # Epanechnikov kernel estimates

        self.bandwidth = (self.T ** (-1/3)) * np.ones(self.N)
        self.calculate_epanechnikov_estimates()

        # colors for various plots; cf. https://xkcd.com/color/rgb/

        self.plot_colors = {
            'sample_bars': 'xkcd:peach',
            'sample_line': 'xkcd:orange',
            'sample_hilbert': 'xkcd:salmon',
            'oracle_mwcv_bars': 'xkcd:sky blue',
            'oracle_mwcv_line': 'xkcd:blue',
            'oracle_mwcv_hilbert': 'xkcd:turquoise',
            'oracle_mwcv_iso_bars': 'xkcd:aqua',
            'oracle_mwcv_iso_line': 'xkcd:teal',
            'oracle_mwcv_iso_hilbert': 'xkcd:cyan',
        }


    def calculate_E(
        self,
        temporal_slice,
        eigensystem=True
        ):
        """
        Calculate the sample estimator E of a part of the data matrix Y
        given by a set of temporal indices specified by the "temporal_slice".
        Calculate moreover (optionally) its eigenvalues and eigenvectors.
        """
        Y_slice = self.Y[:, temporal_slice]
        T_slice = Y_slice.shape[1]

        E = Y_slice @ Y_slice.T / T_slice

        if eigensystem:
            E_eigval, E_eigvec = eigh(E)
            assert np.allclose(
                E,
                E_eigvec @ np.diag(E_eigval) @ E_eigvec.T
            )
            return E, E_eigval, E_eigvec
        else:
            return E
    

    def calculate_E_final(self):
        """
        Calculate the sample estimator E and its eigensystem
        for the temporal slice corresponding to the last T observations.
        """
        self.E, self.E_eigval, self.E_eigvec = self.calculate_E(
            temporal_slice = slice(-self.T,  None)
        )

        if self.n_top is not None:
            self.E_eigval = self.E_eigval[:-self.n_top]

    
    def calculate_oracle_mwcv(self):
        """
        Calculate the oracle estimator's eigenvalues
        via the moving-window cross-validation estimation procedure.
        """
        assert self.T_out is not None

        self.K = int((self.T_total - self.T) / self.T_out)

        self.xi_oracle_mwcv_all = np.zeros((self.K, self.N))

        for mu in range(self.K):

            t_mu = self.T + mu * self.T_out
            
            _, _, E_train_eigvec_mu = self.calculate_E(
                temporal_slice = slice(t_mu - self.T, t_mu)
            )
            
            E_test_mu = self.calculate_E(
                temporal_slice = slice(t_mu, t_mu + self.T_out),
                eigensystem = False
            )
            
            self.xi_oracle_mwcv_all[mu] = [
                E_train_eigvec_mu[:, i] @ E_test_mu @ E_train_eigvec_mu[:, i]
                for i in range(self.N)
            ]

        self.xi_oracle_mwcv = self.xi_oracle_mwcv_all.mean(axis=0)
        self.xi_oracle_mwcv_std = self.xi_oracle_mwcv_all.std(axis=0)

        if self.n_top is not None:
            self.xi_oracle_mwcv = self.xi_oracle_mwcv[:-self.n_top]
            self.xi_oracle_mwcv_std = self.xi_oracle_mwcv_std[:-self.n_top]


    def calculate_oracle_mwcv_iso(self):
        """
        Fit isotonic regression
        (a non-decreasing piecewise-linear real function fitted to a one-dimensional dataset)
        to the (cross-validation-estimated) oracle eigenvalues.
        """
        assert self.T_out is not None

        self.xi_oracle_mwcv_iso = IsotonicRegression().fit_transform(self.E_eigval, self.xi_oracle_mwcv)
    

    def calculate_epanechnikov_estimates(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the sample eigenvalues, as well as (if they are calculated)
        the (cross-validation-estimated) oracle eigenvalues, and their isotonic regression fit.
        """
        if self.n_top is not None:
            self.bandwidth = self.bandwidth[:-self.n_top]

        self.E_eigval_kernel_density, self.E_eigval_kernel_Hilbert = __class__.epanechnikov_estimates(
            x=self.E_eigval,
            bandwidth=self.bandwidth
            )
        
        if self.T_out is not None:
            self.xi_oracle_mwcv_kernel_density, self.xi_oracle_mwcv_kernel_Hilbert = __class__.epanechnikov_estimates(
                x=np.sort(self.xi_oracle_mwcv),
                bandwidth=self.bandwidth
                )
            self.xi_oracle_mwcv_iso_kernel_density, self.xi_oracle_mwcv_iso_kernel_Hilbert = __class__.epanechnikov_estimates(
                x=self.xi_oracle_mwcv_iso,
                bandwidth=self.bandwidth
                )


    @staticmethod
    def epanechnikov(x):
        """
        Calculate the Epanechnikov kernel and its Hilbert transform
        at the elements of a given array x.
        """
        assert isinstance(x, np.ndarray)
        
        y = (3 / (4 * np.sqrt(5))) * (1 - x ** 2 / 5)
        z = np.where(
            np.abs(np.abs(x) - np.sqrt(5)) < 1e-10,
            0.,
            y * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
        )
        
        kernel_density = np.maximum(y, 0)
        kernel_Hilbert = 0.3 * x - z
        
        return kernel_density, kernel_Hilbert
    

    @staticmethod
    def epanechnikov_estimates(x, bandwidth):
        """
        Perform Epanechnikov-kernel estimation
        of the density and Hilbert transform of an array x
        (using the "bandwidth" rescaling factor).
        """
        l1 = x * bandwidth
        l2 = np.array([(x_ - x) / l1 for x_ in x])
        l3, l4 = __class__.epanechnikov(l2)
        
        kernel_density = (l3 / l1).mean(axis=1)
        kernel_Hilbert = (l4 / l1).mean(axis=1)

        return kernel_density, kernel_Hilbert
    

    def rie(
        self,
        x
        ):
        """
        For a given array x of length N,
        calculate the RIE estimator with eigenvalues x
        (its eigenvectors are, by definition, the sample eigenvectors).
        """
        if np.array(x).shape != (self.N,):
            raise Exception('x must be a 1-dim array of the same length as the number of eigenvalues.')
        
        return self.E_eigvec @ np.diag(x) @ self.E_eigvec.T
    

    def frobenius_ratio(self, x, C):
        """
        Calculate the ratio of Frobenius errors:
          - between a RIE estimator with eigenvalues x and the true correlation matrix C,
          - and between the sample estimator E and the true correlation matrix C.
        """
        return __class__.frobenius_norm(self.rie(x) - C) / __class__.frobenius_norm(self.E - C)
    
    
    @staticmethod
    def frobenius_norm(matrix):
        """
        Calculate the Frobenius norm of a given matrix.
        """
        return np.trace(matrix @ matrix.T)


    def hist(
        self,
        show_lambdas=False,
        show_lambdas_density=False,
        show_lambdas_Hilbert=False,
        show_oracle_mwcv=False,
        show_oracle_mwcv_density=False,
        show_oracle_mwcv_Hilbert=False,
        show_oracle_mwcv_iso=False,
        show_oracle_mwcv_iso_density=False,
        show_oracle_mwcv_iso_Hilbert=False,
        bins=None,
        xlim=None,
        ylim=None,
        legend=True,
        savefig=None,
        set_options=True
        ):
        """
        Plot either of the three histograms of:
          - the sample eigenvalues,
          - the (cross-validation-estimated) oracle eigenvalues,
          - and the isotonic regression fit to the latter,
        optionally with their Epanechnikov-kernel-estimated density,
        and/or Hilbert transforms.
        (In other words, any combination of these nine plots.)
        """
        self.bns = self.N // 4 if bins is None else bins

        if show_lambdas:
            plt.hist(
                self.E_eigval,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('sample_bars', 'black'),
                density=True,
                label='sample eigval'
            )

        if show_lambdas_density:
            plt.plot(
                self.E_eigval,
                self.E_eigval_kernel_density,
                color=self.plot_colors.get('sample_line', 'black'),
                label='sample eigval density'
            )
        
        if show_lambdas_Hilbert:
            plt.plot(
                self.E_eigval,
                self.E_eigval_kernel_Hilbert,
                color=self.plot_colors.get('sample_hilbert', 'black'),
                label='sample eigval Hilbert'
            )

        if show_oracle_mwcv:
            assert self.T_out is not None
            plt.hist(
                self.xi_oracle_mwcv,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('oracle_mwcv_bars', 'black'),
                density=True,
                label='oracle eigval'
            )

        if show_oracle_mwcv_density:
            assert self.T_out is not None
            plt.plot(
                self.xi_oracle_mwcv,
                self.xi_oracle_mwcv_kernel_density,
                color=self.plot_colors.get('oracle_mwcv_line', 'black'),
                label='oracle eigval density'
            )
        
        if show_oracle_mwcv_Hilbert:
            assert self.T_out is not None
            plt.plot(
                self.xi_oracle_mwcv,
                self.xi_oracle_mwcv_kernel_Hilbert,
                color=self.plot_colors.get('oracle_mwcv_hilbert', 'black'),
                label='oracle eigval Hilbert'
            )

        if show_oracle_mwcv_iso:
            assert self.T_out is not None
            plt.hist(
                self.xi_oracle_mwcv_iso,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('oracle_mwcv_iso_bars', 'black'),
                density=True,
                label='isotonic oracle eigval'
            )

        if show_oracle_mwcv_iso_density:
            assert self.T_out is not None
            plt.plot(
                self.xi_oracle_mwcv_iso,
                self.xi_oracle_mwcv_iso_kernel_density,
                color=self.plot_colors.get('oracle_mwcv_iso_line', 'black'),
                label='isotonic oracle eigval density'
            )
        
        if show_oracle_mwcv_iso_Hilbert:
            assert self.T_out is not None
            plt.plot(
                self.xi_oracle_mwcv_iso,
                self.xi_oracle_mwcv_iso_kernel_Hilbert,
                color=self.plot_colors.get('oracle_mwcv_iso_hilbert', 'black'),
                label='isotonic oracle eigval Hilbert'
            )

        if set_options:
            plt.xlabel('eigenvalue')
            density_shown = (
                show_lambdas
                or show_lambdas_density
                or show_oracle_mwcv
                or show_oracle_mwcv_density
                or show_oracle_mwcv_iso
                or show_oracle_mwcv_iso_density
            )
            Hilbert_shown = (
                show_lambdas_Hilbert
                or show_oracle_mwcv_Hilbert
                or show_oracle_mwcv_iso_Hilbert
            )
            hist_y_label = ', '.join(
                [
                    label
                    for label, present in zip(
                        ['probability density', 'Hilbert transform'],
                        [density_shown, Hilbert_shown]
                    )
                    if present
                ]
            )
            plt.ylabel(hist_y_label)

            plt.xlim(xlim)
            plt.ylim(ylim)

            if legend:
                plt.legend()
            
            if savefig:
                plt.savefig(fname=savefig)


    def plot(
        self,
        show_oracle_mwcv=False,
        show_oracle_mwcv_errors=False,
        show_oracle_mwcv_errors_every=1,
        show_oracle_mwcv_iso=False,
        xlim=None,
        ylim=None,
        legend=True,
        savefig=None,
        set_options=True
        ):
        """
        Plot some of the following:
          - the (cross-validation-estimated) oracle eigenvalues
            versus the sample eigenvalues;
          - optionally, add error bars from the cross-validation procedure;
          - the isotonic regression fit to the oracle eigenvalues.
        """
        assert self.T_out is not None

        if show_oracle_mwcv:
            if show_oracle_mwcv_errors:
                plt.errorbar(
                    self.E_eigval,
                    self.xi_oracle_mwcv,
                    yerr=self.xi_oracle_mwcv_std,
                    ecolor='gray',
                    errorevery=show_oracle_mwcv_errors_every,
                    color=self.plot_colors.get('oracle_mwcv_bars', 'black'),
                    fmt='^',
                    alpha=0.3,
                    label='oracle eigval'
                )
            else:
                plt.scatter(
                    self.E_eigval,
                    self.xi_oracle_mwcv,
                    color=self.plot_colors.get('oracle_mwcv_bars', 'black'),
                    marker='^',
                    alpha=0.3,
                    label='oracle eigval'
                )

        if show_oracle_mwcv_iso:
            plt.plot(
                self.E_eigval,
                self.xi_oracle_mwcv_iso,
                color=self.plot_colors.get('oracle_mwcv_iso_line', 'black'),
                label='isotonic oracle eigval'
            )

        if set_options:
            plt.xlabel(r'$\lambda$')
            plt.ylabel(r'$\xi$')
            plt.xlim(xlim)
            plt.ylim(ylim)
            if legend:
                plt.legend()
            if savefig:
                plt.savefig(fname=savefig)
