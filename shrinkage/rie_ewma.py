import numpy as np
import matplotlib.pyplot as plt

from .rie_lp import LedoitPecheShrinkage


class EwmaShrinkage(LedoitPecheShrinkage):
    """
    We further build on the Ledoit-Peche shrunk eigenvalues xi^LP_i, i = 1, ..., N,
    and perform nonlinear shrinkage, computing shrunk eigenvalues xi_i,
    in the case of auto-correlations given by the EWMA model.
    """
    def __init__(
        self,
        delta,
        **kwargs
        ):
        super().__init__(name='EWMA', **kwargs)

        self.delta = delta

        self.calculate_xi()
        self.calculate_epanechnikov_estimates_xi()

        self.plot_colors = dict(
            **self.plot_colors,
            **{
                'xi_bars': 'xkcd:ochre',
                'xi_line': 'xkcd:crimson',
                'xi_hilbert': 'xkcd:rust',
            }
        )


    def calculate_xi(self):
        """
        Calculate the EWMA nonlinear shrinkage xi_i
        of the sample eigenvalues lambda_i.
        """
        ed = np.exp(self.delta)
        eda = np.exp(self.delta * self.alpha)
        
        self.xi = (
            (self.E_eigval / self.beta) * eda * (ed - 1.) ** 2 * np.sin(self.delta * self.beta)
            / self.delta
            / ((ed * eda) ** 2 + 1. - 2 * ed * eda * np.cos(self.delta * self.beta))
        )
        
    
    def calculate_epanechnikov_estimates_xi(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the shrunk eigenvalues xi.
        """
        self.xi_kernel_density, self.xi_kernel_Hilbert = __class__.epanechnikov_estimates(
            x=self.xi,
            bandwidth=self.bandwidth
        )