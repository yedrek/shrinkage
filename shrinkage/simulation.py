import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz, sqrtm, eigh, inv
from scipy.stats import ortho_group

from .varma import Varma


class SampleEigenvalues:
    """
    Retrieve a data matrix Y:
      - either generated from a given model,
        based on the true (population) covariance matrix C,
        and the autocorrelation matrix A;
      - or loaded from an external dataset.
    Based on Y:
      - calculate the sample estimator E,
        and its eigensystem, lambda_i and u_i;
      - calculate the oracle eigenvalues,
        via an out-of-sample procedure.
    """
    def __init__(self, method, T, **kwargs):
        self.method = method
        self.T = T
        self.kwargs = kwargs
        if 'T_total' not in self.kwargs:
            self.kwargs['T_total'] = 2 * T
        
        if self.method=='sandwich':
            self.prepare_sandwich()
            self.simulate_Y_sandwich()
        
        elif self.method=='recurrence' and self.kwargs.get('A_model')=='VARMA':
            self.prepare_sandwich()
            self.simulate_Y_recurrence()

        elif self.method=='load':
            self.Y = self.kwargs.get('Y')
            self.N, self.T_total = self.Y.shape
        
        else:
            raise Exception('Unknown method to create the data matrix Y.')

        self.calculate_E_train_eigensystem()

        self.frobenius_C_E = __class__.frobenius_norm(
            self.E - self.population_covariance.C
        ) if self.method != 'load' else None

        if 'T_out' in self.kwargs:
            self.T_out = self.kwargs.get('T_out')
            self.calculate_oracle(T_out=self.T_out)
    

    def prepare_sandwich(self):
        """
        Calculate the cross-covariance matrix C,
        and the auto-covariance matrix A,
        based on given kwargs,
        for the use in a "sandwich" simulation model.
        """
        self.population_covariance = PopulationCovariance(**self.kwargs)
        self.auto_covariance = AutoCovariance(**self.kwargs)

        self.N = self.population_covariance.N
        self.T_total = self.auto_covariance.T_total

    
    def simulate_Y_sandwich(self):
        """
        Simulate synthetic data Y of shape N x T_total
        by first simulating an array X
        of IID random variables from a given distribution,
        then "sandwich"-ing it with square roots of C and A.
        """
        self.dist = self.kwargs.get('dist', 'Gaussian')

        rng = np.random.default_rng()
        if self.dist=='Gaussian':
            X = rng.standard_normal(size=(self.N, self.T_total))
        elif self.dist=='Student-t':
            df = self.kwargs.get('df')
            X = rng.standard_t(df=df, size=(self.N, self.T_total))
        else:
            raise Exception('Unknown distribution.')
        
        self.Y = self.population_covariance.sqrt_C @ X @ self.auto_covariance.sqrt_A
    

    def simulate_Y_recurrence(self):
        """
        Simulate synthetic data Y of shape N x T_total
        by a given recurrence relation.
        """
        if self.A_model=='VARMA':
            warm_start = 1000
            T_full = self.T_total + warm_start * (self.auto_covariance.r1 + self.auto_covariance.r2)

            Y = np.zeros(shape=(self.N, T_full))

            rng = np.random.default_rng()
            if self.dist=='Gaussian':
                eps = self.sqrt_C @ rng.standard_normal(size=(self.N, T_full))
            elif self.dist=='Student-t':
                df = self.kwargs.get('df')
                eps = self.sqrt_C @ rng.standard_t(df=df, size=(self.N, T_full))
            else:
                raise Exception('Unknown distribution.')

            for t in range(self.r2, T_full):
                Y[:, t] = (self.auto_covariance.a_list[::-1] * eps[:, (t - self.auto_covariance.r2):(t + 1)]).sum(axis=1)
            
            for t in range(self.r1, T_full - self.r2):
                Y[:, t] += (self.auto_covariance.b_list[::-1] * Y[:, (t - self.auto_covariance.r1):t]).sum(axis=1)
            
            self.Y = Y[:, -self.T_total:]
        
        else:
            raise Exception('Unknown model to simulate Y by a recurrence relation.')
    

    def calculate_E_train_eigensystem(self):
        """
        Calculate the sample estimator E of the N x T part of the data Y,
        as well as its eigenvalues and eigenvectors.
        (This N x T part of Y is what matters for most of our applications.
        The remaining temporal part is needed only for
        out-of-sample estimation of the oracle estimator.)
        """
        Y_trunc = self.Y[:, :self.T]
        self.E = Y_trunc @ Y_trunc.T / self.T

        self.E_eigval, self.E_eigvec = eigh(self.E)
        
        self.verified_E_eigensystem = np.allclose(
            self.E,
            self.E_eigvec @ np.diag(self.E_eigval) @ self.E_eigvec.T
        )
    

    def rie(self, x):
        """
        For a given array x of length N,
        calculate the RIE estimator with eigenvalues x
        (its eigenvectors are, by definition, the sample eigenvectors u_i).
        """
        if np.array(x).shape != (self.N,):
            raise Exception('x must be a 1-dim array of the same length as the number of eigenvalues.')
        
        return self.E_eigvec @ np.diag(x) @ self.E_eigvec.T
    

    def calculate_oracle(self, T_out):
        """
        Calculate the oracle estimator's eigenvalues via an out-of-sample procedure.
        
        """
        self.n = int((self.T_total - self.T) / T_out)
        if self.n <= 0:
            raise Exception('n must be positive.')

        self.xi_oracle = np.array([
            np.mean(
                [
                    np.mean(
                        (
                            (
                                self.E_eigvec[:, i]
                                @ self.Y[:, (self.T + mu * T_out + 1):(self.T + (mu + 1) * T_out + 1)]
                            )
                        ) ** 2
                    )
                    for mu in np.arange(self.n)
                ]
            )
            for i in np.arange(self.N)
        ])


    def hist(self, show_oracle=False, bins=None, legend=True, savefig=None):
        """
        Plot the histogram of the sample eigenvalues,
        and optionally the oracle eigenvalues.
        """
        bns = self.N // 4 if bins is None else bins
        plt.hist(
            self.E_eigval,
            bins=bns,
            alpha=0.5,
            color='tab:orange',
            density=True,
            label='sample eigval'
        )
        if show_oracle:
            assert self.xi_oracle is not None
            plt.hist(
                self.xi_oracle,
                bins=bns,
                alpha=0.5,
                color='blue',
                density=True,
                label='oracle eigval'
            )

        if legend:
            plt.legend()
        plt.savefig(fname=savefig) if savefig else plt.show()
    

    def frobenius_ratio(self, x):
        """
        Calculate the ratio of Frobenius errors:
          - between a RIE estimator with eigenvalues x and the true correlation matrix C,
          - and between the sample estimator E and the true correlation matrix C.
        """
        return __class__.frobenius_norm(
            self.rie(x) - self.population_covariance.C
        ) / self.frobenius_C_E
    
    
    @staticmethod
    def frobenius_norm(matrix):
        """
        Calculate the Frobenius norm of a given matrix.
        """
        return np.trace(matrix @ matrix.T)


class PopulationCovariance:
    """
    Generate synthetic population ("true") covariance matrix C,
    according to select models.
    """
    def __init__(self, N, C_model='unit', rotate_C=False, **kwargs):
        self.N = N
        self.C_model = C_model
        self.rotate_C = rotate_C

        self.kwargs = kwargs

        self.generate_C()

        self.sqrt_C = sqrtm(self.C).real
    

    def generate_C(self):
        """
        Generate C.
        """

        if self.C_model=='unit':
            self.C = np.ones(self.N)
        
        elif self.C_model=='clusters':
            self.f_list = self.kwargs.get('f_list')
            self.e_list = self.kwargs.get('e_list')

            assert len(self.f_list) == len(self.e_list) - 1
            
            f_list_full = [int(f * self.N) for f in self.f_list]
            f_list_full += [self.N - sum(f_list_full)]
            
            C_list = [s * [e] for s, e in zip(f_list_full, self.e_list)]
            self.C = np.diag(np.array([c for sublist in C_list for c in sublist]))
        
        elif self.C_model=='inverse-Wishart':
            self.kappa = self.kwargs.get('kappa')
            
            q_IW = 1. / (1. + 2 * self.kappa)
            T_IW = int(self.N / q_IW)

            rng = np.random.default_rng()
            R = rng.standard_normal(size=(self.N , T_IW))
            W = R @ R.T / T_IW
            self.C = (1. - q_IW) * inv(W)
        
        elif self.C_model=='Kumaraswamy':
            self.condition_number = self.kwargs.get('condition_number')
            self.a = self.kwargs.get('a')
            self.b = self.kwargs.get('b')
            
            rng = np.random.default_rng()
            kum = (1. - (1. - rng.uniform(size=self.N)) ** (1 / self.b)) ** (1 / self.a)
            C_eigvals = 1. + (self.condition_number - 1.) * kum
            C_eigvals.sort()
            
            self.C = np.diag(C_eigvals)
        
        else:
            raise Exception('Unknown method to generate C.')
        
        if self.rotate_C:
            O = ortho_group.rvs(dim=self.N)
            self.C = O @ self.C @ O.T


class AutoCovariance:
    """
    Generate synthetic auto-covariance matrix A,
    according to select models.
    """
    def __init__(self, T_total, A_model='unit', rotate_A=False, **kwargs):
        self.T_total = T_total
        self.A_model = A_model
        self.rotate_A = rotate_A

        self.kwargs = kwargs

        self.generate_A()

        self.sqrt_A = sqrtm(self.A).real
    

    def generate_A(self):
        """
        Generate A.
        """
        if self.A_model=='unit':
            self.A = np.eye(self.T_total)
        
        elif self.A_model=='VARMA':
            varma = Varma(T=self.T_total, **self.kwargs)
            self.a_list = varma.a_list
            self.b_list = varma.b_list
            self.r1 = varma.r1
            self.r2 = varma.r2
            self.A = varma.A

        elif self.A_model=='exp-decay':
            self.tau = self.kwargs.get('tau')
            self.A = toeplitz(np.exp(- np.arange(self.T_total) / self.tau))
        
        elif self.A_model=='EWMA':
            self.delta = self.kwargs.get('delta')
            eps = 1. - self.delta / self.T_total
            self.A = np.diag(self.T_total * (1 - eps) / (1 - eps ** self.T_total) * eps ** np.arange(self.T_total))
        
        if self.rotate_A:
            O = ortho_group.rvs(dim=self.T_total)
            self.A = O @ self.A @ O.T