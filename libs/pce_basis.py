import numpy as np
import chaospy as cp
from itertools import product
import math
from stlpy.STL import LinearPredicate
import copy


class PCEBasis:

    def __init__(self, eta, q):

        self.eta = eta

        self.basis = cp.generate_expansion(order=q, dist=eta)
        self.nodes, self.weights = cp.generate_quadrature(order=q, dist=eta, rule="Gaussian")
        self.q = q
        self.L = len(self.basis)

        self.psi = np.zeros([self.L, self.L, self.L])

        for s, j, i in product(range(self.L), range(self.L), range(self.L)):
            num = cp.E(self.basis[s]*self.basis[s], eta)
            if abs(num) < 1e-10:
                self.psi[s][j][i] = 0
            else:    
                self.psi[s][j][i] = cp.E(self.basis[s]*self.basis[j]*self.basis[i], eta)/num

    def generate_coefficients_multiple(self, f):

        coeffs = [self.generate_coefficients(fs) for fs in f]

        return np.array(coeffs)

    
    def generate_coefficients(self, f):

        samples = [f(node) for node in self.nodes.T]
        _, coefficients = cp.fit_quadrature(self.basis, self.nodes, self.weights, samples, retall=True)

        return coefficients
    
    def get_mean_from_coef(self, zeta_hat):
        return zeta_hat[0]

    def get_max_coef(self, zeta_hat):

        L = zeta_hat.shape[0]
        S = zeta_hat.shape[1]

        for k in range(1, L):
            for j in range(S):
                zeta_hat[k][j] = zeta_hat[k][j] * np.sqrt(abs(cp.E(self.basis[k] ** 2, self.eta)))

        return np.amax(abs(zeta_hat[1:, :]), axis=0)
    
    def get_coef(self, zeta_hat, row):
        return zeta_hat[row]

    def get_var_from_coef(self, zeta_hat):
        L = zeta_hat.shape[0]
        S = zeta_hat.shape[1]

        Var = [sum([zeta_hat[k][j] ** 2 * cp.E(self.basis[k] ** 2, self.eta) for k in range(1, L)]) for j in range(S)]
        
        return np.array(Var)
    
    def get_std_from_coef(self, zeta_hat):
        L = zeta_hat.shape[0]
        S = zeta_hat.shape[1]

        Var = [sum([zeta_hat[k][j] ** 2 * cp.E(self.basis[k] ** 2, self.eta) for k in range(1, L)]) for j in range(S)]
        
        return np.sqrt(Var)

    
    def gen_bs_predicate(self, alpha, gamma, beta, epsilon, name):

        """
        Create belief-space predicate representing the chance constraint:

        P(gamma'w_t <= beta - alpha'y_t) >= 1 - eps

        :y_t                n dimensional deterministic signal
        :w_t                n dimensional stochastic signal

        This chance constraint can be converted to 

        alpha'y_t + gamma'hat{w}_t^0 ± kappa * sqrt(E(Phi_i^2)) * gamma'\hat{w}_t^i <= beta     for all i=1, 2, ..., L-1

        kappa = sqrt((1-eps)/eps)
        hat{w}_t = {hat{w}_t^0, hat{w}_t^1, ..., hat{w}_{L-1}}:  n * L dimensional PCE coefficients of w_t

        :param gamma:       coefficient vector for stochastic signal w_t (n, )
        :param alpha:        coefficient vector for deterministic signal y_t (n, )
        :param beta:       coefficient scalar (1, )
        :param epsilon:     risk level (1, )

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        kappa = math.sqrt((1 - epsilon)/epsilon)

        coef = kappa * np.array([math.sqrt(abs(cp.E(self.basis[k] ** 2, self.eta))) for k in range(1, self.L)])

        for i in range(1, self.L):
            pre_mat = np.zeros((self.L + 1, alpha.shape[0]))
            pre_mat[0] = alpha
            pre_mat[1] = gamma
            pre_mat[i + 1] = coef[i - 1] * gamma
            formula_p = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), beta, name=name)
            pre_mat[i + 1] = - coef[i - 1] * gamma
            formula_n = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), beta, name=name)

            try:
                formula &= formula_p & formula_n
            except:
                formula = formula_p & formula_n

        return formula