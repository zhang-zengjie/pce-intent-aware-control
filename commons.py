import math
import numpy as np
from itertools import product
import chaospy as cp


base_sampling_time = 0.1
base_length = 5

length = cp.Trunc(cp.Normal(base_length, 0.05), lower=base_length - 0.05, upper=base_length + 0.05)
tau = cp.Trunc(cp.Normal(base_sampling_time, 0.01), lower=base_sampling_time - 0.01, upper=base_sampling_time + 0.01)
eta = cp.J(tau, length)


def gen_linear_matrix(xi_0, u_0):
    theta0, v0 = 0, xi_0[3]
    gamma0 = 0

    A1 = np.array([[0, 0, - v0 * math.sin(theta0 + gamma0), math.cos(theta0 + gamma0)],
                   [0, 0, v0 * math.cos(theta0 + gamma0), math.sin(theta0 + gamma0)],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    A2 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, math.sin(gamma0)],
                   [0, 0, 0, 0]])

    B1 = np.array([[- v0 * math.sin(theta0 + gamma0), 0],
                   [v0 * math.cos(theta0 + gamma0), 0],
                   [0, 0],
                   [0, 1]])

    B2 = np.array([[0, 0],
                   [0, 0],
                   [v0 * math.cos(gamma0), 0],
                   [0, 0]])

    return A1, A2, B1, B2


def gen_linear_scalar(delta_t, l):
    a1 = delta_t
    b1 = delta_t
    a2 = delta_t/l
    b2 = delta_t/l

    return a1, a2, b1, b2


def bicycle_model(xi, u, delta_t, l):
    x, y, theta, v = xi[0], xi[1], xi[2], xi[3]
    gamma, a = u[0], u[1]
    x += delta_t * v * math.cos(theta + gamma)
    y += delta_t * v * math.sin(theta + gamma)
    theta += delta_t * v * math.sin(gamma)/l
    v += delta_t * a
    return x, y, theta, v


def bicycle_linear_model(xi, u, xi_0, u_0, delta_t, l):

    A1, A2, B1, B2 = gen_linear_matrix(xi_0, u_0)
    a1, a2, b1, b2 = gen_linear_scalar(delta_t, l)

    A = a1 * A1 + a2 * A2
    B = b1 * B1 + b2 * B2

    next_xi = xi + np.dot(A, xi) + np.dot(B, u)
    return next_xi


def pce_model(zeta_hat, mu, psi, xi_0, u_0, a_hat):

    zeta_hat_next = zeta_hat
    A1, A2, B1, B2 = gen_linear_matrix(xi_0, u_0)

    a1_hat = a_hat[0]
    a2_hat = a_hat[1]

    b1_hat = a1_hat
    b2_hat = a2_hat

    e1_hat = a1_hat
    e2_hat = a2_hat
    e3_hat = b1_hat
    e4_hat = b2_hat

    E1 = - np.dot(A1, xi_0)
    E2 = - np.dot(A2, xi_0)
    E3 = - np.dot(B1, u_0)
    E4 = - np.dot(B2, u_0)

    for s in range(zeta_hat.shape[0]):

        B = b1_hat[s] * B1 + b2_hat[s] * B2
        E = e1_hat[s] * E1 + e2_hat[s] * E2 + e3_hat[s] * E3 + e4_hat[s] * E4

        zeta_hat_next[s] = zeta_hat[s] + np.dot(B, mu) + E
        for j in range(zeta_hat.shape[1]):
            A = np.inner(a1_hat, psi[s][j]) * A1 + np.inner(a2_hat, psi[s][j]) * A2
            zeta_hat_next[s] += np.dot(A, zeta_hat[j])
    return zeta_hat_next
