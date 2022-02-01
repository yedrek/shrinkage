import numpy as np

from scipy.linalg import toeplitz, sqrtm, inv
from scipy.stats import ortho_group

from .varma import Varma


class PopulationCovariance:
    """
    Generate synthetic population ("true") covariance matrix C,
    according to select models.
    """
    def __init__(
        self,
        N,
        C_model='unit',
        rotate_C=False,
        **kwargs
        ):
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
        
        # optionally, rotate such a generated C by an orthogonal similarity transformation

        if self.rotate_C:
            O = ortho_group.rvs(dim=self.N)
            self.C = O @ self.C @ O.T


class AutoCovariance:
    """
    Generate synthetic auto-covariance matrix A,
    according to select models.
    """
    def __init__(
        self,
        T_total,
        A_model='unit',
        rotate_A=False,
        **kwargs
        ):
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
            varma = Varma(
                T=self.T_total,
                **self.kwargs
                )
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
            self.A = np.diag(
                self.T_total * (1 - eps) / (1 - eps ** self.T_total) * eps ** np.arange(self.T_total)
                )
        
        if self.rotate_A:
            O = ortho_group.rvs(dim=self.T_total)
            self.A = O @ self.A @ O.T


class DataMatrix:
    """
    Retrieve a data matrix Y of shape N x T_total:
      - either generated from a given model,
        based on the true (population) covariance matrix C,
        and the autocorrelation matrix A;
      - or loaded from an external dataset.
    """
    def __init__(
        self,
        method,
        **kwargs
        ):
        self.method = method
        self.kwargs = kwargs

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
