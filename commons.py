import math
import numpy as np
import chaospy as cp
from itertools import product
from stlpy.systems import LinearSystem


def gen_linear_matrix(xi_0):
    theta0, v0 = xi_0[2], xi_0[3]
    gamma0 = 0

    A1 = [[0, 0, - v0 * math.sin(theta0 + gamma0), math.cos(theta0 + gamma0)],
          [0, 0, v0 * math.cos(theta0 + gamma0), math.sin(theta0 + gamma0)],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    A2 = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, math.sin(gamma0)],
          [0, 0, 0, 0]]

    B1 = [[- v0 * math.sin(theta0 + gamma0), 0],
          [v0 * math.cos(theta0 + gamma0), 0],
          [0, 0],
          [0, 1]]

    B2 = [[0, 0],
          [0, 0],
          [v0 * math.cos(gamma0), 0],
          [0, 0]]

    return np.array([A1, A2]), np.array([B1, B2])


def gen_linear_scalar(delta_t, l):
    a1 = delta_t
    b1 = delta_t
    a2 = delta_t/l
    b2 = delta_t/l

    return (a1, a2), (b1, b2)


def bicycle_model(xi, u, delta_t, l):
    x, y, theta, v = xi[0], xi[1], xi[2], xi[3]
    gamma, a = u[0], u[1]
    x += delta_t * v * math.cos(theta + gamma)
    y += delta_t * v * math.sin(theta + gamma)
    theta += delta_t * v * math.sin(gamma)/l
    v += delta_t * a
    return x, y, theta, v


def gen_bicycle_linear_sys(xi_0, delta_t, l):
    A, B = gen_linear_matrix(xi_0)
    a, b = gen_linear_scalar(delta_t, l)

    Am = sum([a[i] * A[i] for i in [0, 1]])
    Bm = sum([b[i] * B[i] for i in [0, 1]])

    n = Bm.shape[0]
    m = Bm.shape[1]

    Cm = np.zeros((m, n))
    Dm = np.zeros((m, m))

    sys = LinearSystem(Am, Bm, Cm, Dm)

    return sys


def bicycle_linear_model(xi, u, xi_0, delta_t, l):

    sys = gen_bicycle_linear_sys(xi_0, delta_t, l)

    next_xi = xi + np.dot(sys.A, xi) + np.dot(sys.B, u)
    return next_xi


def gen_pce_matrix(zeta_hat, psi, xi_0, a_hat):

    A, B = gen_linear_matrix(xi_0)

    b_hat = a_hat

    Bb = np.array([sum([b_hat[i][s] * B[i] for i in [0, 1]])
                   for s in range(zeta_hat.shape[0])])

    Ab = np.array([[sum([np.inner(a_hat[i], psi[s][j]) * A[i] for i in [0, 1]])
                    for j in range(zeta_hat.shape[1])]
                   for s in range(zeta_hat.shape[0])])

    return Ab, Bb


def pce_model(zeta_hat, mu, psi, xi_0, a_hat):

    Ab, Bb = gen_pce_matrix(zeta_hat, psi, xi_0, a_hat)

    zeta_hat_next = np.array([zeta_hat[s] + sum([Ab[s][j] @ zeta_hat[j] for j in range(zeta_hat.shape[1])]) + Bb[s] @ mu for s in range(zeta_hat.shape[0])])

    return zeta_hat_next


def monte_carlo_bicycle(horizon, state_0, control, delta_t, leng):

    samples = np.zeros([horizon + 1, 4])
    samples[0] = state_0
    for k in range(horizon):
        samples[k+1] = bicycle_model(samples[k], control[k], delta_t, leng)
    return samples


def monte_carlo_linear_bicycle(horizon, state_0, control, delta_t, leng):

    samples = np.zeros([horizon + 1, 4])
    samples[0] = state_0
    for k in range(horizon):
        samples[k+1] = bicycle_linear_model(samples[k], control[k], state_0, delta_t, leng)
    return samples


def gen_pce_coefficients(horizon, state_0, control, psi, a_hat):

    L = a_hat.shape[1]
    zeta_hat = np.zeros([horizon + 1, L, 4])
    zeta_hat[0][0] = state_0

    for k in range(horizon):
        zeta_hat[k+1] = pce_model(zeta_hat[k], control[k], psi, state_0, a_hat)

    return zeta_hat
