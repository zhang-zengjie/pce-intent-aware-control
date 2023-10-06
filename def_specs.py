import numpy as np
from stlpy.STL import LinearPredicate
import chaospy as cp
from commons import eta
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
    coef = np.array([math.sqrt((1 - eps) * (L-1) * cp.E(basis[k] ** 2, eta) / eps) for k in range(1, L)])

    d = a.shape[0]

    assert (a.shape[1] == 1), "a must be of shape (d,1)"
    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros((L + 1, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = True

    for i in range(1, L):
        pre_mat[i + 1] = coef[i - 1] * c
        formula &= LinearPredicate(pre_mat.flatten, b)
        pre_mat[i + 1] = - coef[i - 1] * c
        formula &= LinearPredicate(pre_mat.flatten, b)

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

    assert (a.shape[1] == 1), "a must be of shape (d,1)"
    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros((L + 1, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = LinearPredicate(pre_mat.flatten, b)

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
    coef = np.array([math.sqrt((L-1) * cp.E(basis[k] ** 2, eta)) for k in range(1, L)])

    d = c.shape[0]

    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros(L + 1, d)

    formula = True

    for i in range(L-1):
        pre_mat[i + 1] = coef[i - 1] * c
        formula &= LinearPredicate(pre_mat.flatten, b)
        pre_mat[i + 1] = - coef[i - 1] * c
        formula &= LinearPredicate(pre_mat.flatten, b)

    return formula