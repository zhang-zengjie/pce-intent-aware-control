import numpy as np
from stlpy.STL import LinearPredicate
import chaospy as cp
from gen_basis import eta
import numpoly
import math

basis = numpoly.load('basis.npy')


def probability_formula(a, c, b, eps, name=None):
    """
    Create STL formulas representing the chance constraint:

    P(a'x_t + c'z_t >= b) >= 1 - eps

    This chance constraint can be converted to 

    a'x_t + c'z_t^0 ± coef_i c'z_t^i -b >=0     for all i=1, 2, ..., L-1
    given coef_i = sqrt((1-eps)/eps * (L-1) * E(Phi_i^2))

    :param a:           coefficient vector (n, )
    :param c:           coefficient vector (n, )
    :param b:           coefficient scalar (1, )
    :param x_t:         the n dimensional system state at time t (deterministic)
    :param z_t:         the n dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold (1, )

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = basis.shape[0]
    coef = np.array([math.sqrt((1 - eps) * (L-1) * abs(cp.E(basis[k] ** 2, eta)) / eps) for k in range(1, L)])

    n = a.shape[0]

    pre_mat = np.zeros((L + 1, n))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = True

    for i in range(1, L):
        pre_mat[i + 1] = coef[i - 1] * c
        formula_p = LinearPredicate(pre_mat.reshape((1, -1)), b)
        pre_mat[i + 1] = - coef[i - 1] * c
        formula_n = LinearPredicate(pre_mat.reshape((1, -1)), b)

        try:
            formula &= formula_p & formula_n
        except:
            formula = formula_p & formula_n

    return formula

def expecation_formula(a, c, b, name=None):
    """
    Create STL formulas representing the expectation constraint:

    a'x_t + E(c'z_t) >= b

    This chance constraint can be converted to 

    a'x_t + c'z_t^0 >= b

    :param a:           coefficient vector 
    :param c:           coefficient vector 
    :param b:           coefficient scalar
    :param x_t:         the d dimensional system state at time t (deterministic)
    :param z_t:         the d dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = basis.shape[0]
    d = a.shape[0]

    pre_mat = np.zeros((L + 1, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = LinearPredicate(pre_mat.reshape((1, -1)), b)

    return formula

def variance_formula(c, b, name=None):
    """
    Create STL formulas representing the expectation constraint:

    Var(c'z_t) <= b^2

    This chance constraint can be converted to 

    -b <= coef_i c'z_t^i <= b     for all i=1, 2, ..., L-1
    given coef_i = sqrt((L-1) * E(Phi_i^2))

    :param a:           coefficient vector 
    :param c:           coefficient vector 
    :param b:           coefficient scalar
    :param x_t:         the d dimensional system state at time t (deterministic)
    :param z_t:         the d dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = basis.shape[0]
    coef = np.array([math.sqrt((L-1) * abs(cp.E(basis[k] ** 2, eta))) for k in range(1, L)])

    n = c.shape[0]

    pre_mat = np.zeros((L + 1, n))

    for i in range(L-1):
        pre_mat[i + 1] = coef[i - 1] * c
        formula_p = LinearPredicate(pre_mat.reshape((1, -1)), -b)
        pre_mat[i + 1] = - coef[i - 1] * c
        formula_n = LinearPredicate(pre_mat.reshape((1, -1)), b)

        try:
            formula &= formula_p & formula_n
        except:
            formula = formula_p & formula_n

    return formula

def neg_variance_formula(c, b, name=None):
    """
    Create STL formulas representing the expectation constraint:

    Var(c'z_t) >= b^2

    This chance constraint can be converted to 

    coef_i c'z_t^i >= b | coef_i c'z_t^i <= -b     for all i=1, 2, ..., L-1
    given coef_i = sqrt((L-1) * E(Phi_i^2))

    :param a:           coefficient vector 
    :param c:           coefficient vector 
    :param b:           coefficient scalar
    :param x_t:         the d dimensional system state at time t (deterministic)
    :param z_t:         the d dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = basis.shape[0]
    coef = np.array([math.sqrt((L-1) * abs(cp.E(basis[k] ** 2, eta))) for k in range(1, L)])

    n = c.shape[0]

    pre_mat = np.zeros((L + 1, n))

    for i in range(L-1):
        pre_mat[i + 1] = coef[i - 1] * c
        formula_p = LinearPredicate(pre_mat.reshape((1, -1)), b)
        pre_mat[i + 1] = - coef[i - 1] * c
        formula_n = LinearPredicate(pre_mat.reshape((1, -1)), b)

        try:
            formula &= formula_p | formula_n
        except:
            formula = formula_p | formula_n
        
    return formula