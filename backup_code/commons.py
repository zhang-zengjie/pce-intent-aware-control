import math
import numpy as np
import chaospy as cp
from itertools import product


base_sampling_time = 0.1
base_length = 5

length = cp.Trunc(cp.Normal(base_length, 0.05), lower=base_length - 0.05, upper=base_length + 0.05)
tau = cp.Trunc(cp.Normal(base_sampling_time, 0.01), lower=base_sampling_time - 0.01, upper=base_sampling_time + 0.01)
eta = cp.J(tau, length)


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


def bicycle_linear_model(xi, u, xi_0, delta_t, l):

    A, B = gen_linear_matrix(xi_0)
    a, b = gen_linear_scalar(delta_t, l)

    Am = sum([a[i] * A[i] for i in [0, 1]])
    Bm = sum([b[i] * B[i] for i in [0, 1]])

    next_xi = xi + np.dot(Am, xi) + np.dot(Bm, u)
    return next_xi


def bicycle_forward(horizon, state_0, control, delta_t, l):

    samples = np.zeros([horizon + 1, 4])
    samples[0] = state_0
    for k in range(horizon):
        samples[k+1] = bicycle_model(samples[k], control[k], delta_t, l)
    return samples


def bicycle_linear_forward(horizon, state_0, control, delta_t, l):

    samples = np.zeros([horizon + 1, 4])
    samples[0] = state_0
    for k in range(horizon):
        samples[k+1] = bicycle_linear_model(samples[k], control[k], state_0, delta_t, l)
    return samples


def gen_pce_coefficients(horizon, state_0, control, psi, a_hat):

    pce_system = PCESystem(state_0, a_hat, psi)

    L = a_hat.shape[1]
    zeta_hat = np.zeros([horizon + 1, L, 4])
    zeta_hat[0][0] = state_0

    pce_system.update_matrices(state_0)

    for k in range(horizon):
        zeta_hat[k+1] = pce_system.pce_model(zeta_hat[k], control[k])

    return zeta_hat


class PCESystem:
    """
    A linear discrete-time system of the coupled form

    .. math::

        x_{t+1, s} = sum_s A_s x_{t,s} + B_s u_t

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param A: A ``(n,n)`` numpy array representing the state transition matrix
    :param B: A ``(n,m)`` numpy array representing the control input matrix
    :param C: A ``(p,n)`` numpy array representing the state output matrix
    :param D: A ``(p,m)`` numpy array representing the control output matrix
    """
    def __init__(self, xi_0, a_hat, psi):

        self.Ab = None
        self.Bb = None
        self.a_hat = a_hat
        self.psi = psi
        self.update_matrices(xi_0)

    def update_matrices(self, xi_0):

        A, B = gen_linear_matrix(xi_0)

        b_hat = self.a_hat

        self.Bb = np.array([sum([b_hat[i][s] * B[i] for i in [0, 1]])
                            for s in range(self.a_hat.shape[1])])

        self.Ab = np.array([[sum([np.inner(self.a_hat[i], self.psi[s][j]) * A[i] for i in [0, 1]])
                             for j in range(xi_0.shape[0])]
                            for s in range(self.a_hat.shape[1])])

    def pce_model(self, state, mu):

        return np.array([state[s] + sum([np.dot(self.Ab[s][j], state[j]) for j in range(state.shape[1])])
                         + np.dot(self.Bb[s], mu) for s in range(state.shape[0])])
