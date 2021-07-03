import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import nroots, re, im, symbols, sympify
from types import FunctionType
from collections import defaultdict
from tqdm.auto import tqdm

from .varma import Varma


class LedoitPecheShrinkage:
    """
    Given sample eigenvalues lambda_i, i = 1, ..., N, of an estimator E,
    as well as the number of samples T:
      - estimate the eigenvalue density and Hilbert transform of E
        via the (adaptive) Epanechnikov kernel
        [Ledoit, O., Wolf, M., Analytical nonlinear shrinkage of large-dimensional covariance matrices];
      - calculate the Ledoit-Peche shrunk eigenvalues xi_i, i = 1, ..., N.
    """
    def __init__(self, lambdas, T):
        """
        Instantiated with a 1-dimensional array of eigenvalues, lambdas, and a number of samples, T.
        Calculates:
          - N = number of eigenvalues (length of lambdas);
          - q = N/T;
          - rescaling factor "bandwidth" for kernel estimation; set to T^(-1/3) for each eigenvalue;
          - Epanechnikov kernel estimation of the density and Hilbert transform of lambdas;
          - intermediate and final Ledoit-Peche variables alpha_i, beta_i, u_i, xi_LP_i;
          - Epanechnikov kernel estimation of the density and Hilbert transform of xi_LP_i.
        """
        self.lambdas = np.array(lambdas)
        self.T = T
        
        self.N = len(self.lambdas)
        self.q = self.N / self.T
        
        self.bandwidth = (self.T ** (-1/3)) * np.ones(self.N)

        self.name = 'Ledoit-Peche'
        
        self._calculate_epanechnikov_estimates(eigval='lambdas')
        self._calculate_LP_variables()
        self._calculate_epanechnikov_estimates(eigval='xi_LP')

        self.batch_idx = None
        self.N_batch = None

        self.chi_roots_branch = None
        self.xi_branch = None
        self.M_of_N_branch = None
        self.n_branches = None
        self.xi = None
        self.chi_roots = None
        self.chi_grads = None
        self.xi_grads = None

        self.xi_kernel_density = None
        self.xi_kernel_Hilbert = None


    def _calculate_epanechnikov_estimates(self, eigval='lambdas'):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of a given array "eigval", be it "lambdas", "xi_LP" or "xi".
        """
        if eigval=='lambdas':
            self.lambdas_kernel_density, self.lambdas_kernel_Hilbert = __class__.epanechnikov_estimates(
                x=self.lambdas,
                bandwidth=self.bandwidth
            )
        
        elif eigval=='xi_LP':
            self.xi_LP_kernel_density, self.xi_LP_kernel_Hilbert = __class__.epanechnikov_estimates(
                x=self.xi_LP,
                bandwidth=self.bandwidth
            )
        
        elif eigval=='xi':
            if self.xi is None or len(self.xi) < self.N:
                self.predict()
            self.xi_kernel_density, self.xi_kernel_Hilbert = __class__.epanechnikov_estimates(
                x=self.xi,
                bandwidth=self.bandwidth
            )
        
        else:
            raise Exception('The argument "eigval" must be either "lambdas", "xi_LP", or "xi".')
    

    def _calculate_LP_variables(self):
        """
        Calculate the intermediate variables alpha_i and beta_i,
        plus the complex numbers u_i = alpha_i + i * beta_i,
        of the Ledoit-Peche nonlinear shrinkage xi_i = xi_LP_i, which also compute here,
        of the sample eigenvalues lambda_i.
        """
        self.alpha = self.q * (self.lambdas * self.lambdas_kernel_Hilbert - 1.)
        self.beta = np.pi * self.q * self.lambdas * self.lambdas_kernel_density

        self.u_range = np.array([complex(al, be) for al, be in zip(self.alpha, self.beta)])

        self.xi_LP = self.lambdas / ((self.alpha + 1.) ** 2 + self.beta ** 2)
    

    def _set_batch_idx(self, batch_idx=None):
        """
        Choose indices i, from 0 to (N - 1), on which to perform the calculation of xi_i.
        By default, all the indices are chosen, i.e. we calculate for every i = 0, ..., N - 1.
        """
        self.batch_idx = list(range(self.N)) if batch_idx is None else batch_idx
        self.N_batch = len(self.batch_idx)

    def predict(self, batch_idx=None):
        """
        Calculate the Ledoit-Peche nonlinear shrinkage xi_i
        of the sample eigenvalues lambda_i.
        Note: We've already calculated xi_LP_i = xi_i, so this is a simple substitution.
        A "predict" method is needed for consistency here and in every child class.
        """
        self._set_batch_idx(batch_idx=batch_idx)
        self.xi = self.xi_LP[self.batch_idx]

        self.xi_branch = [self.xi]
        self.n_branches = len(self.xi_branch)
    

    def hist(
        self,
        show_lambdas=False,
        show_lambdas_density=False,
        show_lambdas_Hilbert=False,
        show_xi=False,
        show_xi_density=False,
        show_xi_Hilbert=False,
        show_xi_LP=False,
        show_xi_LP_density=False,
        show_xi_LP_Hilbert=False,
        bins=None,
        xlim=None,
        ylim=None,
        legend=True,
        savefig=None
    ):
        """
        Plot three histograms:
          - of the sample eigenvalues lambda_i,
          - the Ledoit-Peche shrunk eigenvalues xi_LP_i,
          - and the shrunk eigenvalues xi_i,
        optionally with their Epanechnikov-kernel-estimated density,
        and/or Hilbert transforms.
        (In other words, any combination of these nine plots.)
        """
        bns = self.N // 4 if bins is None else bins
        if show_lambdas:
            plt.hist(
                self.lambdas,
                bins=bns,
                alpha=0.5,
                color='tab:orange',
                density=True,
                label='sample eigval'
            )
        
        if show_lambdas_density:
            plt.plot(
                self.lambdas,
                self.lambdas_kernel_density,
                color='tab:red',
                label='sample eigval density'
            )
        
        if show_lambdas_Hilbert:
            plt.plot(
                self.lambdas,
                self.lambdas_kernel_Hilbert,
                color='tab:green',
                label='sample eigval Hilbert'
            )
        
        if show_xi_LP:
            plt.hist(
                self.xi_LP,
                bins=bns,
                alpha=0.5,
                color='tab:pink',
                density=True,
                label=f'Ledoit-Peche shrunk eigval'
            )
        
        if show_xi_LP_density:
            plt.plot(
                self.xi_LP,
                self.xi_LP_kernel_density,
                color='brown',
                label=f'Ledoit-Peche shrunk eigval density'
            )
        
        if show_xi_LP_Hilbert:
            plt.plot(
                self.xi_LP,
                self.xi_LP_kernel_Hilbert,
                color='tab:cyan',
                label=f'Ledoit-Peche shrunk eigval Hilbert'
            )
        
        if show_xi:
            if self.xi is None or len(self.xi) < self.N:
                self.predict()
            plt.hist(
                self.xi,
                bins=bns,
                alpha=0.5,
                color='fuchsia',
                density=True,
                label=f'{self.name} shrunk eigval'
            )
        
        if show_xi_density:
            if self.xi_kernel_density is None:
                self._calculate_epanechnikov_estimates(eigval='xi')
            plt.plot(
                self.xi,
                self.xi_kernel_density,
                color='purple',
                label=f'{self.name} shrunk eigval density'
            )
        
        if show_xi_Hilbert:
            if self.xi_kernel_Hilbert is None:
                self._calculate_epanechnikov_estimates(eigval='xi')
            plt.plot(
                self.xi,
                self.xi_kernel_Hilbert,
                color='tab:olive',
                label=f'{self.name} shrunk eigval Hilbert'
            )

        plt.xlim(xlim)
        plt.ylim(ylim)
        if legend:
            plt.legend()
        plt.savefig(fname=savefig) if savefig else plt.show()


    def plot(self, branch=None, xlim=None, ylim=None, legend=True, savefig=None):
        """
        Plot the shrunk eigenvalues xi_i (vertical axis)
        versus the sample eigenvalues lambda_i (horizontal axis).
        Often, there are multiple solutions for xi_i (branches), stored in "xi_branch",
        and you can plot all or some of them here, besides the correct branch xi_i:
        set "branch" to either "all", or a string number of a branch (one-indexed),
        or a list of such string numbers.
        """
        if self.xi is None or len(self.xi) < self.N:
            self.predict()
        
        if branch is None:
            plt.plot(
                self.lambdas,
                self.xi,
                color='purple',
                label=f'{self.name} shrunk vs. sample eigval'
            )

        else:
            if branch=='all':
                branches_to_plot = range(self.n_branches)
            elif isinstance(branch, list):
                branches_to_plot = [int(br) - 1 for br in branch]
            elif isinstance(branch, str):
                branches_to_plot = [int(branch) - 1]
            
            for br in branches_to_plot:
                plt.plot(
                    self.lambdas,
                    self.xi_branch[br],
                    color=cm.hot(0.2 + 0.6 * br / self.n_branches),
                    label=f'{self.name} shrunk (branch {br + 1}) vs. sample eigval'
                )

        plt.xlabel('lambda')
        plt.ylabel('xi')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if legend:
            plt.legend()
        plt.savefig(fname=savefig) if savefig else plt.show()
    

    def plot_with_oracle(self, xi_oracle, show_LP=False, xlim=None, ylim=None, legend=True, savefig=None):
        """
        Plot the shrunk eigenvalues xi_i (vertical axis)
        versus the sample eigenvalues lambda_i (horizontal axis).
        Optionally, make an analogous plot of the Ledoit-Peche xi_LP_i versus lambda_i.
        Moreover, make a scatter plot of the oracle eigenvalues xi_oracle_i versus lambda_i.
        """
        plt.plot(
            self.lambdas,
            self.xi,
            color='purple',
            label=f'{self.name} shrunk vs. sample eigval'
        )
        if show_LP:
            plt.plot(
                self.lambdas,
                self.xi_LP,
                color='brown',
                label=f'Ledoit-Peche shrunk vs. sample eigval'
            )
        plt.scatter(
            self.lambdas,
            xi_oracle,
            marker='^',
            alpha=0.3,
            label='oracle eigval'
        )

        plt.xlabel('lambda')
        plt.ylabel('xi')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if legend:
            plt.legend()
        plt.savefig(fname=savefig) if savefig else plt.show()
    

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


class EwmaShrinkage(LedoitPecheShrinkage):
    """
    Given sample eigenvalues lambda_i, i = 1, ..., N, of an estimator E,
    as well as the number of samples T,
    perform nonlinear shrinkage, computing shrunk eigenvalues xi_i,
    in the case of auto-correlations given by the EWMA model.
    """
    def __init__(self, lambdas, T, delta):
        super().__init__(lambdas=lambdas, T=T)
        self.name = 'EWMA'
        self.delta = delta


    def predict(self):
        """
        Calculate the EWMA nonlinear shrinkage xi_i
        of the sample eigenvalues lambda_i.
        """
        if self.alpha is None or self.beta is None:
            super().calculate_LP_variables()
        
        ed = np.exp(self.delta)
        eda = np.exp(self.delta * self.alpha)
        
        self.xi = (
            (self.lambdas / self.beta) * eda * (ed - 1.) ** 2 * np.sin(self.delta * self.beta)
            / self.delta
            / ((ed * eda) ** 2 + 1. - 2 * ed * eda * np.cos(self.delta * self.beta))
        )

        self.xi_branch = [self.xi]
        self.n_branches = len(self.xi_branch)


class VarmaShrinkage(LedoitPecheShrinkage, Varma):
    """
    Given sample eigenvalues lambda_i, i = 1, ..., N, of an estimator E,
    as well as the number of samples T,
    perform nonlinear shrinkage, computing shrunk eigenvalues xi_i,
    in the case of auto-correlations given by the VARMA model.
    """
    def __init__(self, lambdas, T):
        """
        "Adapt" the model to the sample eigenvalues lambdas (and to T).
        """
        LedoitPecheShrinkage.__init__(self, lambdas=lambdas, T=T)


    def set_params(self, **kwargs):
        """
        Set the model's parameters: either tau or a_list and b_list,
        by initializing the parent class handling the parameters;
        it also calculates the autocorrelation matrix A.
        """
        Varma.__init__(self, T=self.T, get_chi_equations=True, **kwargs)


    def get_params(self):
        """
        Return a dictionary with the model's parameters.
        """
        return {
            'a_list': self.a_list,
            'b_list': self.b_list
        }


    def predict(self, batch_idx=None, calculate_grads=False):
        """
        Calculate the VARMA nonlinear shrinkage xi_i
        of the sample eigenvalues lambda_i,
        for several solvable cases.
        """
        self._solve_chi_equation(
            batch_idx=batch_idx,
            calculate_grads=calculate_grads
        )
    
    
    def _solve_chi_equation(self, batch_idx=None, calculate_grads=False):
        """
        For each value of u_i = alpha_i + i * beta_i, for i = 1, ..., N,
        solve an algebraic equation pol = 0 in the variable chi,
        where "pol" will come from a separate file with VARMA polynomials,
        with coefficients depending on u_i, as well as the model's parameters.
        Take the imaginary part of each solution,
        and collect them in branches of the shrunk eigenvalues xi_i.
        Moreover, create a list of the values of M_A(1/chi) for each u_i,
        which should be equal to u_i on the correct branch.
        (Recall, M_A(z) is the M-transform of A,
        and chi = chi_A(z) = 1/N_A(z) is the chi-transform of A.)
        """
        self._set_batch_idx(batch_idx=batch_idx)

        u_batch = self.u_range[self.batch_idx]

        # solve the fundamental polynomial equation in chi at each u in the batch
        # (which will produce n_branches solutions at each u)

        chi = symbols('chi')

        self.chi_roots_branch = []
        chi_roots_im_branch = []
        self.M_of_N_branch = []

        for u in u_batch:
            chi_roots = nroots(
                self.pol(sympify(u), chi, *self.a_list, *self.b_list)
            )
            self.chi_roots_branch.append(chi_roots)

            chi_roots_im = [
                float(im(chi_root))
                for chi_root in chi_roots
            ]
            chi_roots_im_branch.append(chi_roots_im)

            M_of_N = [
                self.calculate_M_transform_A(
                    z_re=float(re(1 / chi_root)),
                    z_im=float(im(1 / chi_root)),
                    method='eig'
                )
                for chi_root in chi_roots
            ]
            self.M_of_N_branch.append(M_of_N)
        
        # reshape to (n_branches, N_batch)
        
        self.chi_roots_branch = np.array(self.chi_roots_branch).T
        prefactor = self.lambdas[self.batch_idx] / self.beta[self.batch_idx]
        self.xi_branch = prefactor * np.array(chi_roots_im_branch).T
        self.M_of_N_branch = np.array(self.M_of_N_branch).T

        self.n_branches = len(self.xi_branch)

        # sort the branches according to xi, for convenience

        sort_idx = np.argsort(self.xi_branch, axis=0)

        self.chi_roots_branch = np.take_along_axis(self.chi_roots_branch, sort_idx, axis=0)
        self.xi_branch = np.take_along_axis(self.xi_branch, sort_idx, axis=0)
        self.M_of_N_branch = np.take_along_axis(self.M_of_N_branch, sort_idx, axis=0)

        # choose one "good" branch xi, by which we mean the one on which M_A(N_A(u)) = u
        # these are now 1D arrays of length N_batch

        sort_idx = np.argsort(np.abs(self.M_of_N_branch - u_batch), axis=0)

        self.chi_roots = np.take_along_axis(self.chi_roots_branch, sort_idx, axis=0)[0]
        self.xi = np.take_along_axis(self.xi_branch, sort_idx, axis=0)[0]
        
        # calculate gradients of the solution chi (also the shrunk eigenvalues xi) w.r.t. VARMA parameters

        if calculate_grads:
            self.chi_grads = defaultdict(list)
            for u, chi in zip(u_batch, self.chi_roots):
                pol_grad_chi = self.pol_grads['chi'](sympify(u), chi, *self.a_list, *self.b_list)
                for param in self.ab:
                    self.chi_grads[param].append(
                        (- self.pol_grads[param](sympify(u), chi, *self.a_list, *self.b_list) / pol_grad_chi).evalf()
                    )
            self.chi_grads = dict(self.chi_grads)

            self.xi_grads = {}
            for param in self.ab:
                self.xi_grads[param] = prefactor * np.array([
                    float(im(chi_grad)) for chi_grad in self.chi_grads[param]
                ])
    

    def fit(
        self,
        xi_oracle,
        loss='mse',
        loss_grad=None,
        optimizer='brute',
        **kwargs
    ):
        """
        Find the VARMA parameters
        for which an error (specified by the loss function)
        between the shrunk eigenvalues and the given oracle eigenvalues
        is minimal.
        Use one of several provided optimization methods ("optimizer").
        """
        self._set_loss(loss=loss, loss_grad=loss_grad)

        self.loss_list = []
        
        if optimizer=='brute':
            self.grid = kwargs.get('grid')

            for params_dict in tqdm(self.grid):
                self.set_params(**params_dict)
                self.predict()

                self.loss_list.append(
                    np.mean(self.loss(xi_oracle, self.xi))
                )
        
        elif optimizer=='gd':
            lr = kwargs.get('lr')
            n_epochs = kwargs.get('n_epochs')
            N_batch = kwargs.get('N_batch')
            n_batches = int(self.N // N_batch)
            r1 = kwargs.get('r1', None)
            r2 = kwargs.get('r2', None)

            self._set_random_params(r1, r2)

            self.grid = []
            rng = np.random.default_rng()
            for epoch in range(n_epochs):
                print(f'Epoch {epoch} commencing...')

                for _ in tqdm(range(n_batches)):
                    params_dict = self.get_params()

                    batch_idx=rng.integers(low=0, high=self.N, size=N_batch) if N_batch < self.N else None

                    self.predict(batch_idx=batch_idx, calculate_grads=True)

                    self.loss_grads = [
                        np.mean(
                            self.loss_grad(xi_oracle[self.batch_idx], self.xi)
                            * self.xi_grads[param]
                        )
                        for param in self.ab
                    ]

                    self.set_params(
                        a_list=[
                            a_prev - lr * gd
                            for a_prev, gd in zip(params_dict['a_list'], self.loss_grads[:(self.r2 + 1)])
                        ],
                        b_list=[
                            b_prev - lr * gd
                            for b_prev, gd in zip(params_dict['b_list'], self.loss_grads[(self.r2 + 1):])
                        ]
                    )
                
                print('Making prediction on the whole dataset...')

                self.grid.append(params_dict)
                self.predict()
                self.loss_list.append(
                    np.mean(self.loss(xi_oracle, self.xi))
                )

                print('... done.')

        idx_best = np.argmin(self.loss_list)
        self.params_dict_best = self.grid[idx_best]
        self.set_params(**self.params_dict_best)
        self.predict(calculate_grads=True)
        self.loss_best = self.loss_list[idx_best]
        self.loss_LP = np.mean(self.loss(xi_oracle, self.xi_LP))
    

    def _set_loss(self, loss, loss_grad=None):
        """
        Set the loss function (and its gradient w.r.t. xi_pred),
        being a function of xi_true and xi_pred,
        based on the "loss" argument.
        """
        if type(loss)==FunctionType:
            self.loss = loss
            self.loss_grad = loss_grad
        elif loss=='mse':
            self.loss = lambda xi_true, xi_pred: (xi_true - xi_pred) ** 2
            self.loss_grad = lambda xi_true, xi_pred: -2 * (xi_true - xi_pred)
        else:
            raise Exception('Unknown error function.')
    

    def _set_random_params(self, r1=None, r2=None):
        rng = np.random.default_rng()
        if r1 is None and r2 is None:
            self.set_params(
                tau=rng.random()
            )
        else:
            eps = 0.1
            random_list = list(eps * rng.random(size=(r1 + r2 + 1)))
            self.set_params(
                a_list=[1. - random_list[0]] + random_list[1:(r2 + 1)],
                b_list=random_list[(r2 + 1):]
            )