import numpy as np
from scipy.linalg import toeplitz, eigh, inv
from pathlib import Path
from sympy import *

CHI_EQUATIONS_DIR = 'chi_equations'


class Varma:
    """
    Handles the parameters and autocorrelation matrix A
    of a VARMA(r1, r2) model,
    as well as polynomial equations for chi = chi_A(u),
    when they are provided in appropriate text files.
    """
    def __init__(self, T, get_chi_equations=False, **kwargs):
        self.T = T
        self.get_chi_equations = get_chi_equations
        self.kwargs = kwargs

        self._get_varma_parameters()
        self._calculate_A()
        if self.get_chi_equations:
            self._get_chi_equation()


    def _get_varma_parameters(self):
        """
        From the kwargs, retrieve the a- and b-parameters
        of a general VARMA(r1, r1) model.
        Also, set the model's name appropriately.
        """
        if 'tau' in self.kwargs:
            self.tau = self.kwargs.get('tau')
            b1 = np.exp(-1. / self.tau)
            self.a_list = np.array([np.sqrt(1. - b1 ** 2)])
            self.b_list = np.array([b1])
        else:
            self.a_list = np.array(self.kwargs.get('a_list', [1.]))
            self.b_list = np.array(self.kwargs.get('b_list', []))
            self.tau = None

        self.r1 = len(self.b_list)
        self.r2 = len(self.a_list) - 1

        assert self.r2 >= 0

        if self.r1==0:
            self.name = f'VMA({self.r2})'
            self.chi_equations_path = Path('shrinkage') / CHI_EQUATIONS_DIR / f'vma_{self.r2}'
        elif self.r2==0:
            self.name = f'VAR({self.r1})'
            self.chi_equations_path = Path('shrinkage') / CHI_EQUATIONS_DIR / f'var_{self.r1}'
        else:
            self.name = f'VARMA({self.r1}, {self.r2})'
            self.chi_equations_path = Path('shrinkage') / CHI_EQUATIONS_DIR / f'varma_{self.r1}_{self.r2}'
        
        self.ab = [f'a{i}' for i in range(self.r2 + 1)] + [f'b{i}' for i in range(1, self.r1 + 1)]
    

    def _calculate_A(self):
        """
        Calculate the autocorrelation matrix A
        of the full VARMA(r1, r2) model,
        as well as its eigenvalues.
        """
        A_VMA = __class__._calculate_A_vma(
            a_list=self.a_list,
            T=self.T
        )
        A_VMA_2 = __class__._calculate_A_vma(
            a_list=[1.] + [-b for b in self.b_list],
            T=self.T
        )
        self.A = A_VMA @ inv(A_VMA_2)

        self.A_eigval, _ = eigh(self.A)

    
    def calculate_M_transform_A(self, z_re, z_im, method='eig'):
        """
        Calculate the M-transform M_A(z)
        at complex argument z = z_re + i * z_im
        of the autocorrelation matrix A.
        """
        z = complex(z_re, z_im)

        if method=='inv':
            g = np.trace(inv(z * np.eye(self.T) - self.A)) / self.T
            return z * g - 1.
        elif method=='eig':
            g = (1. / (z - self.A_eigval)).mean()
            return z * g - 1.
        else:
            raise Exception('Unknown method of calculating M_A(z).')


    def _get_chi_equation(self):
        """
        Retrieve text files from an appropriate directory
        that contain the polynomial equation for chi = chi_A(u),
        as well as polynomial expressions for the gradients
        of this equation w.r.t. chi and the VARMA parameters.
        Read the polynomials from the text files,
        convert them to sympy expressions, then further to lambda functions,
        with arguments u, chi, and the VARMA parameters.
        """
        if self.chi_equations_path.is_dir():
            params = ['chi'] + self.ab
            args = symbols(' '.join(['u'] + params))

            with open(self.chi_equations_path / 'pol.txt', 'r') as text_file:
                pol_sympy = sympify(text_file.read())
            self.pol = lambdify(args, pol_sympy)
            
            self.pol_grads = {}
            for param in params:
                with open(self.chi_equations_path / f'grad_{param}.txt', 'r') as text_file:
                    pol_grad_sympy = sympify(text_file.read())
                self.pol_grads[param] = lambdify(args, pol_grad_sympy)

        else:
            raise Exception('Equation for this model is not provided in an appropriate text file.')


    @staticmethod
    def _calculate_A_vma(a_list, T):
        """
        Calculate the autocorrelation matrix A
        of a VMA(r2) model.
        """
        r2 = len(a_list) - 1
        kappa_list = [
            sum(
                a_list[j] * a_list[j + i]
                for j in range(r2 - i + 1)
            )
            for i in range(r2 + 1)
        ]
        return toeplitz(kappa_list + [0] * (T - r2 - 1))

# run this once (py -m varma)
# in order to create a directory with text files
# containing polynomials in chi = chi_A(u)
# (unless they already exist)

if __name__ == '__main__':
    Path(CHI_EQUATIONS_DIR).mkdir(parents=True, exist_ok=True)

    a0, a1, a2, b1, b2, k0, k1, k2, A, B, C, N, chi, u = symbols('a0 a1 a2 b1 b2 k0 k1 k2 A B C N chi u')

    def write_to_file(pol_sympy, dir_name, file_name):
        if pol_sympy.is_polynomial():
            file_path = Path(CHI_EQUATIONS_DIR) / dir_name / f'{file_name}.txt'
            with open(file_path, 'w') as f:
                f.write(str(pol_sympy))
            print(f'Written {file_name} polynomial to file.')
        else:
            print(f'Error! The function provided for {file_name} is not a polynomial.')
    
    def write_to_dir(pol_sympy, dir_name, params):
        (Path(CHI_EQUATIONS_DIR) / dir_name).mkdir(parents=True, exist_ok=True)
        
        write_to_file(
            pol_sympy=pol_sympy,
            dir_name=dir_name,
            file_name='pol'
        )
        for param in params:
            write_to_file(
                pol_sympy=collect(diff(pol_sympy, param), chi),
                dir_name=dir_name,
                file_name=f'grad_{param}'
            )

    # VMA(1)

    vma_1 = collect(
        expand(
            (1 - (a0 + a1) ** 2 * chi) * (1 - (a0 - a1) ** 2 * chi) * (1 + u) ** 2 - 1
        ), chi
    )

    write_to_dir(
        pol_sympy=vma_1,
        dir_name='vma_1',
        params=[a0, a1, chi]
    )

    # VAR(1)

    var_1_subs = [
        (a0, 1 / a0),
        (a1, - b1 / a0)
    ]

    var_1 = collect(
        factor(
            vma_1.subs(
                [(chi, 1 / chi), (u, - u - 1)] + var_1_subs
            )
        ) * chi ** 2 * a0 ** 4, chi
    )
    
    write_to_dir(
        pol_sympy=var_1,
        dir_name='var_1',
        params=[a0, b1, chi]
    )

    # VARMA(1, 1)

    varma_1_1 = collect(
        factor(
            ((1 - b1) ** 2 - (a0 + a1) ** 2 * chi)
            * ((1 + b1) ** 2 - (a0 - a1) ** 2 * chi)
            * (b1 * u + a0 * a1 * (1 + u) * chi) ** 2
            - (a0 * a1 * (1 + b1 ** 2) + (a0 ** 2 + a1 ** 2) * b1) ** 2 * chi ** 2
        ) / (chi * a0 * a1 + b1), chi
    )
    
    write_to_dir(
        pol_sympy=varma_1_1,
        dir_name='varma_1_1',
        params=[a0, a1, b1, chi]
    )

    # VMA(2)

    vma_2_pre = (
        ( (1 + u) ** 2 * (N + A) * (N + B) * (N + C) - N ** 2 * (N + (A + B)/2) ) ** 2
        - (N + A) * (N + B) * (N ** 2 - (1 + u) ** 2 * (N + A) * (N + B)) ** 2
    )
    vma_2_subs = [
        (A, -k0 + 2 * k1 - 2 * k2),
        (B, -k0 - 2 * k1 - 2 * k2),
        (C, -k0 + 6 * k2),
        (k0, a0 ** 2 + a1 ** 2 + a2 ** 2),
        (k1, a0 * a1 + a1 * a2),
        (k2, a0 * a2)
    ]

    vma_2 = collect(
        expand(
            chi ** 5 * vma_2_pre.subs(
                [(N , 1 / chi)] + vma_2_subs
            )
        ), chi
    )

    write_to_dir(
        pol_sympy=vma_2,
        dir_name='vma_2',
        params=[a0, a1, a2, chi]
    )

    # VAR(2)

    var_2_subs = var_1_subs + [(a2, - b2 / a0)]

    var_2 = collect(
        expand(
            vma_2_pre.subs(
                [(N , chi), (u, - u - 1)] + vma_2_subs + var_2_subs
            ) * a0 ** 12 / 4
        ), chi
    )
    
    write_to_dir(
        pol_sympy=var_2,
        dir_name='var_2',
        params=[a0, b1, b2, chi]
    )