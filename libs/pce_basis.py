import numpy as np
import chaospy as cp
from itertools import product


class PCEBasis:

    def __init__(self, eta, q):

        self.eta = eta

        self.basis = cp.generate_expansion(order=q, dist=eta)
        self.nodes, self.weights = cp.generate_quadrature(order=q, dist=eta, rule="Gaussian")
        self.q = q
        self.L = len(self.basis)

        self.psi = np.zeros([self.L, self.L, self.L])

        for s, j, i in product(range(self.L), range(self.L), range(self.L)):
            self.psi[s][j][i] = cp.E(self.basis[s]*self.basis[j]*self.basis[i], eta)/cp.E(self.basis[s]*self.basis[s], eta)

    def generate_coefficients_multiple(self, f):

        coeffs = [self.generate_coefficients(fs) for fs in f]

        return np.array(coeffs)

    
    def generate_coefficients(self, f):

        try:
            assert len(f) == 1
        except:
            print("Function 'generate_coefficients' is only for single functions. For multiple functions, use 'generate_coefficients_multiple' instead.")

        samples = [f(node) for node in self.nodes.T]
        _, coefficients = cp.fit_quadrature(self.basis, self.nodes, self.weights, samples, retall=True)

        return coefficients