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

        samples = [f(node) for node in self.nodes.T]
        _, coefficients = cp.fit_quadrature(self.basis, self.nodes, self.weights, samples, retall=True)

        return coefficients
    
    def get_mean_from_coef(self, zeta_hat):
        N = len(zeta_hat)
        E = [zeta_hat[i][0] for i in range(N)]
        return np.array(E)


    def get_var_from_coef(self, zeta_hat):
        N = zeta_hat.shape[0]
        L = zeta_hat.shape[1]
        S = zeta_hat.shape[2]
        '''
        Var = np.zeros([N, 4])
        for i, j in product(range(N), range(S)):
            Var[i][j] = sum([zeta_hat[i][k][j] ** 2 * cp.E(self.basis[k] ** 2, self.eta) for k in range(1, L)])
        '''
        Var = [[sum([zeta_hat[i][k][j] ** 2 * cp.E(self.basis[k] ** 2, self.eta) for k in range(1, L)])
                      for j in range(S) ] for i in range(N)]
        return np.array(Var)