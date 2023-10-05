import numpy as np
from stlpy.STL import LinearPredicate

def probability_formula(a, c, b, eps, Phi, name=None):
    """
    Create STL formulas representing the chance constraint:

    P(a'x_t + c'z_t >= b) >= 1 - eps

    This chance constraint can be converted to 

    a'x_t + c'z_t^0 ± coef_i z_t^i -b >=0     for all i=1, 2, ..., L-1
    given coef_i = sqrt((1-eps)/eps) * (L-1) * sqrt(E(Phi_i^2))

    :param a:           coefficient vector 
    :param c:           coefficient vector 
    :param b:           coefficient scalar
    :param x_t:         the d dimensional system state at time t (deterministic)
    :param z_t:         the d dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = coef.shape[0]
    d = a.shape[0]
    e = np.ones((d, ))

    assert (a.shape[1] == 1), "a must be of shape (d,1)"
    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros((L, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = True

    for i in range(L-1):
        pre_mat[i+1] = coef[i+1] * e
        formula &= LinearPredicate(pre_mat.flatten, b)
        pre_mat[i+1] = - coef[i+1] * e
        formula &= LinearPredicate(pre_mat.flatten, b)

    return formula

def expecation_formula(a, c, b, Psi, name=None):
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

    L = coef.shape[0]
    d = a.shape[0]
    e = np.ones((d, ))

    assert (a.shape[1] == 1), "a must be of shape (d,1)"
    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros((L, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = LinearPredicate(pre_mat.flatten, b)

    return formula

def variant_formula(a, c, b, Psi, name=None):
    """
    Create STL formulas representing the expectation constraint:

    Var(c'z_t) <= b^2

    This chance constraint can be converted to 

    -b <= coef_i z_t^i <= b     for all i=1, 2, ..., L-1
    given coef_i = sqrt((1-eps)/eps) * (L-1) * sqrt(E(Phi_i^2))

    :param a:           coefficient vector 
    :param c:           coefficient vector 
    :param b:           coefficient scalar
    :param x_t:         the d dimensional system state at time t (deterministic)
    :param z_t:         the d dimensional system state at time t (stochastic)
    :param eps:         probabilistic threshold

    :return formula:    An ``STLFormula`` specifying the converted deterministic
                        specifications.
    """

    L = coef.shape[0]
    d = a.shape[0]
    e = np.ones((d, ))

    assert (a.shape[1] == 1), "a must be of shape (d,1)"
    assert (c.shape[1] == 1), "c must be of shape (d,1)"
    assert (b.shape == (1,)), "b must be of shape (1,)"

    pre_mat = np.zeros((L, d))
    pre_mat[0] = a
    pre_mat[1] = c

    formula = True

    for i in range(L-1):
        pre_mat[i+1] = coef[i+1] * e
        formula &= LinearPredicate(pre_mat.flatten, b)
        pre_mat[i+1] = - coef[i+1] * e
        formula &= LinearPredicate(pre_mat.flatten, b)

    return formula