import numpy as np
import matplotlib.pyplot as plt

from .sample import SampleEigenvalues


class LedoitPecheShrinkage(SampleEigenvalues):
    """
    We further build on the sample eigenvalues lambda_i, i = 1, ..., N,
    to calculate the Ledoit-Peche shrunk eigenvalues xi^LP_i, i = 1, ..., N.
    """
    def __init__(
        self,
        tau_eff_list = None,
        **kwargs
        ):

        super().__init__(name='Ledoit-Peche', **kwargs)

        self.calculate_LP_variables()
        self.calculate_epanechnikov_estimates_xi_LP()

        if tau_eff_list is not None:
            self.fit_T_eff(tau_eff_list=tau_eff_list)
            self.calculate_epanechnikov_estimates_xi_LP_eff_best_oracle_mwcv()

        self.plot_colors = dict(
            **self.plot_colors,
            **{
                'xi_LP_bars': 'xkcd:pale pink',
                'xi_LP_line': 'xkcd:pale purple',
                'xi_LP_hilbert': 'xkcd:light purple',
                'xi_LP_eff_best_oracle_mwcv_bars': 'xkcd:lilac',
                'xi_LP_eff_best_oracle_mwcv_line': 'xkcd:violet',
                'xi_LP_eff_best_oracle_mwcv_hilbert': 'xkcd:light violet',
            }
        )


    def calculate_LP_variables_any_q(self, q):
        """
        Calculate the intermediate variables alpha_i and beta_i,
        plus the complex numbers u_i = alpha_i + i * beta_i,
        appearing in the Ledoit-Peche nonlinear shrinkage formula for xi_LP_i,
        corresponding to any sample eigenvalue lambda_i.
        Importantly, keep q variable at this stage.
        """
        alpha = q * (self.E_eigval * self.E_eigval_kernel_Hilbert - 1.)
        beta = np.pi * q * self.E_eigval * self.E_eigval_kernel_density

        u_range = np.array([complex(al, be) for al, be in zip(alpha, beta)])

        xi_LP = self.E_eigval / ((alpha + 1.) ** 2 + beta ** 2)

        return alpha, beta, u_range, xi_LP


    def calculate_LP_variables(self):
        """
        Calculate the Ledoit-Peche variables for the actual q of the system.
        """
        self.alpha, self.beta, self.u_range, self.xi_LP = self.calculate_LP_variables_any_q(q=self.q)


    def fit_T_eff(
        self,
        tau_eff_list
        ):
        """
        Choose an "effective" q = N/T_eff for which the Ledoit-Peche shrunk eigenvalues
        lie closest (in terms of the MSE) to the (cross-validation-estimated) oracle eigenvalues.
        """
        assert self.T_out is not None

        def T_eff_from_tau(tau):
            return self.T * (1. - np.exp(-1. / tau))

        self.loss_list_xi_LP_eff_oracle_mwcv = []
        xi_LP_eff_list = []
        for tau in tau_eff_list:
            T_eff = T_eff_from_tau(tau)
            q_eff = self.N / T_eff
            _, _, _, xi_LP_eff = self.calculate_LP_variables_any_q(q=q_eff)

            xi_LP_eff_list.append(xi_LP_eff)

            self.loss_list_xi_LP_eff_oracle_mwcv.append(
                np.mean((xi_LP_eff - self.xi_oracle_mwcv) ** 2)
            )
        
        idx_best = np.argmin(self.loss_list_xi_LP_eff_oracle_mwcv)
        self.tau_eff_best_oracle_mwcv = tau_eff_list[idx_best]
        self.T_eff_best_oracle_mwcv = T_eff_from_tau(self.tau_eff_best_oracle_mwcv)
        self.xi_LP_eff_best_oracle_mwcv = xi_LP_eff_list[idx_best]


    def calculate_epanechnikov_estimates_xi_LP(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the Ledoit-Peche shrunk eigenvalues xi_LP.
        """
        self.xi_LP_kernel_density, self.xi_LP_kernel_Hilbert = __class__.epanechnikov_estimates(
            x=self.xi_LP,
            bandwidth=self.bandwidth
        )
    

    def calculate_epanechnikov_estimates_xi_LP_eff_best_oracle_mwcv(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the Ledoit-Peche shrunk eigenvalues xi_LP..
        """
        self.xi_LP_eff_best_oracle_mwcv_kernel_density, self.xi_LP_eff_best_oracle_mwcv_kernel_Hilbert = \
        __class__.epanechnikov_estimates(
            x=self.xi_LP_eff_best_oracle_mwcv,
            bandwidth=self.bandwidth
        )
    

    def hist(
        self,
        show_xi_LP=False,
        show_xi_LP_density=False,
        show_xi_LP_Hilbert=False,
        show_xi_LP_eff_best_oracle_mwcv=False,
        show_xi_LP_eff_best_oracle_mwcv_density=False,
        show_xi_LP_eff_best_oracle_mwcv_Hilbert=False,
        savefig=None,
        set_options=True,
        **kwargs
    ):
        """
        Add another two histograms:
          - of the Ledoit-Peche shrunk eigenvalues xi_LP_i,
          - and the same, but with the best effective q_eff.
        """
        super().hist(**kwargs, savefig=None, set_options=False)

        if show_xi_LP:
            plt.hist(
                self.xi_LP,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('xi_LP_bars', 'black'),
                density=True,
                label='Ledoit-Peche shrunk eigval'
            )

        if show_xi_LP_density:
            plt.plot(
                self.xi_LP,
                self.xi_LP_kernel_density,
                color=self.plot_colors.get('xi_LP_line', 'black'),
                label='Ledoit-Peche shrunk eigval density'
            )
        
        if show_xi_LP_Hilbert:
            plt.plot(
                self.xi_LP,
                self.xi_LP_kernel_Hilbert,
                color=self.plot_colors.get('xi_LP_hilbert', 'black'),
                label='Ledoit-Peche shrunk eigval Hilbert'
            )

        if show_xi_LP_eff_best_oracle_mwcv:
            plt.hist(
                self.xi_LP_eff_best_oracle_mwcv,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('xi_LP_eff_best_oracle_mwcv_bars', 'black'),
                density=True,
                label='effective-T Ledoit-Peche shrunk eigval'
            )

        if show_xi_LP_eff_best_oracle_mwcv_density:
            plt.plot(
                self.xi_LP_eff_best_oracle_mwcv,
                self.xi_LP_eff_best_oracle_mwcv_kernel_density,
                color=self.plot_colors.get('xi_LP_eff_best_oracle_mwcv_line', 'black'),
                label='effective-T Ledoit-Peche shrunk eigval density'
            )
        
        if show_xi_LP_eff_best_oracle_mwcv_Hilbert:
            plt.plot(
                self.xi_LP_eff_best_oracle_mwcv,
                self.xi_LP_eff_best_oracle_mwcv_kernel_Hilbert,
                color=self.plot_colors.get('xi_LP_eff_best_oracle_mwcv_hilbert', 'black'),
                label='effective-T Ledoit-Peche shrunk eigval Hilbert'
            )
        
        if set_options:
            plt.xlabel('eigenvalue')
            density_shown = (
                kwargs.get('show_lambdas', False)
                or kwargs.get('show_lambdas_density', False)
                or kwargs.get('show_oracle_mwcv', False)
                or kwargs.get('show_oracle_mwcv_density', False)
                or kwargs.get('show_oracle_mwcv_iso', False)
                or kwargs.get('show_oracle_mwcv_iso_density', False)
                or show_xi_LP
                or show_xi_LP_density
                or show_xi_LP_eff_best_oracle_mwcv
                or show_xi_LP_eff_best_oracle_mwcv_density
            )
            Hilbert_shown = (
                kwargs.get('show_lambdas_Hilbert', False)
                or kwargs.get('show_oracle_mwcv_Hilbert', False)
                or kwargs.get('show_oracle_mwcv_iso_Hilbert', False)
                or show_xi_LP_Hilbert
                or show_xi_LP_eff_best_oracle_mwcv_Hilbert
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

            plt.xlim(kwargs.get('xlim', None))
            plt.ylim(kwargs.get('ylim', None))

            if kwargs.get('legend', True):
                plt.legend()
            
            if savefig:
                plt.savefig(fname=savefig)


    def plot(
        self,
        show_xi_LP=False,
        show_xi_LP_eff_best_oracle_mwcv=False,
        savefig=None,
        set_options=True,
        **kwargs
        ):
        """
        Add to the plot some of these graphs:
          - the Ledoit-Peche shrunk eigenvalues, with the actual value of q;
          - the same but with the best effective q_eff.
        """
        super().plot(savefig=None, set_options=False, **kwargs)

        if show_xi_LP:
            plt.plot(
                self.E_eigval,
                self.xi_LP,
                color=self.plot_colors.get('xi_LP_line', 'black'),
                label='Ledoit-Peche shrunk eigval'
            )
        
        if show_xi_LP_eff_best_oracle_mwcv:
            plt.plot(
                self.E_eigval,
                self.xi_LP_eff_best_oracle_mwcv,
                color=self.plot_colors.get('xi_LP_eff_best_oracle_mwcv_line', 'black'),
                label='effective-T Ledoit-Peche shrunk eigval'
            )
        
        if set_options:
            plt.xlabel(r'$\lambda$')
            plt.ylabel(r'$\xi$')
            plt.xlim(kwargs.get('xlim', None))
            plt.ylim(kwargs.get('ylim', None))
            if kwargs.get('legend', True):
                plt.legend()
            if savefig:
                plt.savefig(fname=savefig)

