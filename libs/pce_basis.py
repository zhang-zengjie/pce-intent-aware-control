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
            self.psi[s][j][i] = cp.E(self.basis[s]*self.basis[j]*self.basis[i], eta)/cp.E(self.basis[s]*self.basis[s], eta)

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
    
    def probability_formula(self, a, c, b, eps, name):

        """
        Create STL formulas representing the chance constraint:

        P(a'x_t + c'z_t >= b) >= 1 - eps

        :x_t                n dimensional deterministic signal
        :z_t                n dimensional stochastic signal

        This chance constraint can be converted to 

        a'x_t + c'hat{z}_t^0 ± coef_i c'\hat{z}_t^i >= b     for all i=1, 2, ..., L-1
        given coef_i = sqrt((1-eps)/eps * (L-1) * E(Phi_i^2))  for all i=1, 2, ..., L-1

        hat{z}_t:           n * L dimensional PCE coefficients of z_t

        :param a:           coefficient vector (n, )
        :param c:           coefficient vector (n * L, )
        :param b:           coefficient scalar (1, )
        :param eps:         probabilistic threshold (1, )

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        coef = np.array([math.sqrt((1 - eps) * (self.L-1) * abs(cp.E(self.basis[k] ** 2, self.eta)) / eps) for k in range(1, self.L)])
        # coef = np.sqrt([(self.L - 1) * abs(cp.E(self.basis[k] ** 2, self.eta)) for k in range(1, self.L)])

        for i in range(1, self.L):
            pre_mat = np.zeros((self.L + 1, a.shape[0]))
            pre_mat[0] = a
            pre_mat[1] = c
            pre_mat[i + 1] = coef[i - 1] * c
            formula_p = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), b, name=name)
            pre_mat[i + 1] = - coef[i - 1] * c
            formula_n = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), b, name=name)

            try:
                formula &= formula_p & formula_n
            except:
                formula = formula_p & formula_n

        return formula

    def ExpectationPredicate(self, beta, alpha, gamma, name):

        """
        Create STL formulas representing the expectation constraint:

        beta'y_t + E(alpha'w_t) >= gamma

        :y_t                n dimensional deterministic signal
        :w_t                n dimensional stochastic signal

        This chance constraint can be converted to 

        beta'y_t + alpha'hat{w}_t^0 >= b

        :param beta:           coefficient vector (n, )
        :param alpha:           coefficient vector (n, )
        :param gamma:           coefficient scalar (1, )

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        pre_mat = np.zeros((self.L + 1, beta.shape[0]))
        pre_mat[0] = beta
        pre_mat[1] = alpha

        formula = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), gamma, name=name)

        return formula

    def variance_formula(self, c, b, name):

        """
        Create STL formulas representing the expectation constraint:

        Var(c'z_t) <= b^2

        :z_t                n dimensional stochastic signal
        :c                  n dimensional selection vector

        This chance constraint can be converted to 

        -b <= coef_i c'z_t^i <= b     for all i=1, 2, ..., L-1
        given coef_i = sqrt((L-1) * E(Phi_i^2))     for all i=1, 2, ..., L-1

        :param c:           n dimensional coefficient vector 
        :param b:           coefficient scalar

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        coef = np.sqrt([(self.L - 1) * abs(cp.E(self.basis[k] ** 2, self.eta)) for k in range(1, self.L)])

        

        for i in range(1, self.L):
            pre_mat = np.zeros((self.L + 1, c.shape[0]))
            pre_mat[i + 1] = coef[i - 1] * c
            formula_p = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), -b, name=name)
            pre_mat[i + 1] = - coef[i - 1] * c
            formula_n = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), -b, name=name)

            try:
                formula &= formula_p & formula_n
            except:
                formula = formula_p & formula_n

        return formula

    def neg_variance_formula(self, c, b, name):

        """
        Create STL formulas representing the expectation constraint:

        Var(c'z_t) >= b^2

        :z_t                n dimensional stochastic signal

        This chance constraint can be converted to 

        coef_i c'z_t^i >= b | coef_i c'z_t^i <= -b     for all i=1, 2, ..., L-1
        given coef_i = sqrt((L-1) * E(Phi_i^2))     for all i=1, 2, ..., L-1

        :param c:           n dimentional coefficient vector 
        :param b:           coefficient scalar

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        coef = np.sqrt([(self.L - 1) * abs(cp.E(self.basis[k] ** 2, self.eta)) for k in range(1, self.L)])

        for i in range(1, self.L):
            pre_mat = np.zeros((self.L + 1, c.shape[0]))
            pre_mat[i + 1] = coef[i - 1] * c
            formula_p = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), b, name=name)
            pre_mat[i + 1] = - coef[i - 1] * c
            formula_n = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), b, name=name)

            try:
                formula |= formula_p | formula_n
            except:
                formula = formula_p | formula_n
            
        return formula
    
    def BeliefSpacePredicate(self, beta, alpha, gamma, epsilon, name):

        """
        Create belief-space predicate representing the chance constraint:

        P(alpha'w_t <= gamma - beta'y_t) >= 1 - eps

        :y_t                n dimensional deterministic signal
        :w_t                n dimensional stochastic signal

        This chance constraint can be converted to 

        beta'y_t + alpha'hat{w}_t^0 ± kappa * sqrt(E(Phi_i^2)) * alpha'\hat{w}_t^i <= gamma     for all i=1, 2, ..., L-1

        kappa = sqrt((1-eps)/eps)
        hat{w}_t = {hat{w}_t^0, hat{w}_t^1, ..., hat{w}_{L-1}}:  n * L dimensional PCE coefficients of w_t

        :param alpha:       coefficient vector for stochastic signal w_t (n, )
        :param beta:        coefficient vector for deterministic signal y_t (n, )
        :param gamma:       coefficient scalar (1, )
        :param epsilon:     risk level (1, )

        :return formula:    An ``STLFormula`` specifying the converted deterministic
                            specifications.
        """

        kappa = math.sqrt((1 - epsilon)/epsilon)

        coef = kappa * np.array([math.sqrt(abs(cp.E(self.basis[k] ** 2, self.eta))) for k in range(1, self.L)])

        for i in range(1, self.L):
            pre_mat = np.zeros((self.L + 1, beta.shape[0]))
            pre_mat[0] = beta
            pre_mat[1] = alpha
            pre_mat[i + 1] = coef[i - 1] * alpha
            formula_p = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), gamma, name=name)
            pre_mat[i + 1] = - coef[i - 1] * alpha
            formula_n = LinearPredicate(copy.copy(pre_mat.reshape((1, -1))), gamma, name=name)

            try:
                formula &= formula_p & formula_n
            except:
                formula = formula_p & formula_n

        return formula