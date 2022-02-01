import numpy as np
import matplotlib.pyplot as plt
from sympy import nroots, re, im, symbols, sympify
from types import FunctionType
from collections import defaultdict
from tqdm.auto import tqdm

from .varma import Varma
from .rie_lp import LedoitPecheShrinkage


class VarmaShrinkage(LedoitPecheShrinkage, Varma):
    """
    We further build on the Ledoit-Peche shrunk eigenvalues xi^LP_i, i = 1, ..., N,
    and perform nonlinear shrinkage, computing shrunk eigenvalues xi_i,
    in the case of auto-correlations given by the VARMA model.
    """
    def __init__(
        self,
        **kwargs
        ):
        
        LedoitPecheShrinkage.__init__(
            self,
            **kwargs
            )
        
        self.plot_colors = dict(
            **self.plot_colors,
            **{
                'xi_bars': 'xkcd:mauve',
                'xi_line': 'xkcd:maroon',
                'xi_hilbert': 'xkcd:rose',
            }
        )
    

    def set_params(
        self,
        **kwargs
        ):
        """
        Set the VARMA model's parameters: either tau or a_list and b_list,
        by initializing the parent class handling the parameters;
        it also calculates the autocorrelation matrix A.
        """
        Varma.__init__(
            self,
            T=self.T,
            get_chi_equations=True,
            **kwargs
            )
    

    def get_params(self):
        """
        Return a dictionary with the model's parameters.
        """
        return {
            'a_list': self.a_list,
            'b_list': self.b_list
        }


    def calculate_xi(
        self,
        batch_idx=None,
        calculate_grads=False,
        calculate_epanechnikov_estimates_xi=False
        ):
        """
        Calculate the VARMA nonlinear shrinkage xi_i
        of the sample eigenvalues lambda_i,
        for several solvable cases.
        """
        self.solve_chi_equation(
            batch_idx=batch_idx,
            calculate_grads=calculate_grads
        )

        if calculate_epanechnikov_estimates_xi:
            self.calculate_epanechnikov_estimates_xi()
    
    
    def calculate_epanechnikov_estimates_xi(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the shrunk eigenvalues xi.
        """
        self.xi_kernel_density, self.xi_kernel_Hilbert = __class__.epanechnikov_estimates(
            x=self.xi,
            bandwidth=self.bandwidth
        )
    

    def set_batch_idx(
        self,
        batch_idx=None
        ):
        """
        Choose indices i, from 0 to (N - 1), on which to perform the calculation of xi_i.
        By default, all the indices are chosen, i.e. we calculate for every i = 0, ..., N - 1.
        """
        n_top_actual = self.n_top if self.n_top is not None else 0
        self.batch_idx = list(range(self.N - n_top_actual)) if batch_idx is None else batch_idx
        self.N_batch = len(self.batch_idx)


    def solve_chi_equation(
        self,
        batch_idx=None,
        calculate_grads=False
        ):
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
        self.set_batch_idx(batch_idx=batch_idx)

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
        prefactor = self.E_eigval[self.batch_idx] / self.beta[self.batch_idx]
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
    

    def fit_params(
        self,
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
        self.set_loss(
            loss=loss,
            loss_grad=loss_grad
            )

        self.loss_list_xi_oracle_mwcv = []
        
        if optimizer=='brute':
            self.grid = kwargs.get('grid')

            for params_dict in tqdm(self.grid):
                self.set_params(**params_dict)
                self.calculate_xi()

                self.loss_list_xi_oracle_mwcv.append(
                    np.mean(self.loss(self.xi_oracle_mwcv, self.xi))
                )
        
        elif optimizer=='gd':
            lr = kwargs.get('lr')
            n_epochs = kwargs.get('n_epochs')
            N_batch = kwargs.get('N_batch')
            n_batches = int(self.N // N_batch)
            r1 = kwargs.get('r1', None)
            r2 = kwargs.get('r2', None)

            self.set_random_params(r1, r2)

            self.grid = []
            rng = np.random.default_rng()
            for epoch in range(n_epochs):
                print(f'Epoch {epoch} commencing...')

                for _ in tqdm(range(n_batches)):
                    params_dict = self.get_params()

                    batch_idx=rng.integers(low=0, high=self.N, size=N_batch) if N_batch < self.N else None

                    self.calculate_xi(batch_idx=batch_idx, calculate_grads=True)

                    self.loss_grads = [
                        np.mean(
                            self.loss_grad(self.xi_oracle_mwcv[self.batch_idx], self.xi)
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
                self.calculate_xi()
                self.loss_list_xi_oracle_mwcv.append(
                    np.mean(self.loss(self.xi_oracle_mwcv, self.xi))
                )

                print('... done.')

        idx_best = np.argmin(self.loss_list_xi_oracle_mwcv)
        self.params_dict_best = self.grid[idx_best]
        self.set_params(**self.params_dict_best)
        self.calculate_xi(
            calculate_grads=True,
            calculate_epanechnikov_estimates_xi=True
            )
        self.loss_best = self.loss_list_xi_oracle_mwcv[idx_best]
    

    def set_loss(
        self,
        loss,
        loss_grad=None
        ):
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
    

    def set_random_params(
        self,
        r1=None,
        r2=None
        ):
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
    

    def hist(
        self,
        show_xi=False,
        show_xi_density=False,
        show_xi_Hilbert=False,
        savefig=None,
        set_options=True,
        **kwargs
    ):
        """
        Add another one histogram of the shrunk eigenvalues xi_i.
        """
        LedoitPecheShrinkage.hist(self, savefig=None, set_options=False, **kwargs)

        if show_xi:
            plt.hist(
                self.xi,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('xi_bars', 'black'),
                density=True,
                label=f'{self.name} shrunk eigval'
            )

        if show_xi_density:
            plt.plot(
                self.xi,
                self.xi_kernel_density,
                color=self.plot_colors.get('xi_line', 'black'),
                label=f'{self.name} shrunk eigval density'
            )
        
        if show_xi_Hilbert:
            plt.plot(
                self.xi,
                self.xi_kernel_Hilbert,
                color=self.plot_colors.get('xi_hilbert', 'black'),
                label=f'{self.name} shrunk eigval Hilbert'
            )
        
        if set_options:
            plt.xlim(kwargs.get('xlim', None))
            plt.ylim(kwargs.get('ylim', None))
            if kwargs.get('legend', True):
                plt.legend()
            if savefig:
                plt.savefig(fname=savefig)


    def plot(
        self,
        show_xi=False,
        savefig=None,
        set_options=True,
        **kwargs
        ):
        """
        Add to the plot a graph of the shrunk eigenvalues.
        """
        LedoitPecheShrinkage.plot(self, savefig=None, set_options=False, **kwargs)

        if show_xi:
            plt.plot(
                self.E_eigval,
                self.xi,
                color=self.plot_colors.get('xi_line', 'black'),
                label=f'{self.name} shrunk eigval'
            )
        
        if set_options:
            plt.xlabel('lambda')
            plt.ylabel('xi')
            plt.xlim(kwargs.get('xlim', None))
            plt.ylim(kwargs.get('ylim', None))
            if kwargs.get('legend', True):
                plt.legend()
            if savefig:
                plt.savefig(fname=savefig)
