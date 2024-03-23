from stlpy.systems import NonlinearSystem, LinearSystem
import numpy as np
import math


def get_linear_matrix(x_curr, dt):
    theta0, v0 = x_curr[2], x_curr[3]
    gamma0 = 0
    
    A0 = [[0, 0, - v0 * math.sin(theta0 + gamma0), math.cos(theta0 + gamma0)],
          [0, 0, v0 * math.cos(theta0 + gamma0), math.sin(theta0 + gamma0)],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    A1 = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, math.sin(gamma0)],
          [0, 0, 0, 0]]

    B0 = [[- v0 * math.sin(theta0 + gamma0), 0],
          [v0 * math.cos(theta0 + gamma0), 0],
          [0, 0],
          [0, 1]]

    B1 = [[0, 0],
          [0, 0],
          [v0 * math.cos(gamma0), 0],
          [0, 0]]
    
    E0 = [v0 * math.sin(theta0 + gamma0) * theta0, - v0 * math.cos(theta0 + gamma0) * theta0, 0, 0]

    E1 = [0, 0, 0, 1]

    return np.array([A0, A1]) * dt, np.array([B0, B1]) * dt, np.array([E0, E1]) * dt


class BicycleModel(NonlinearSystem):

    def __init__(self, dt, x0, param, N, intent_gain=1, intent_offset=0, pce=False, Q=None, R=None, useq=None, basis=None, name=None, color=None):

        self.n = 4
        self.m = 2
        self.p = 4
        self.name = name
        self.useq = useq

        self.Q = Q
        self.R = R
        self.N = N
        self.intent_gain = intent_gain
        self.intent_offset = intent_offset

        self.dt = dt
        self.color = color
        self.param = param
        self.PCE = pce

        # Param list: bias (delta), length (l), intent (iota)
        self.fn = [
            lambda z: z[0],             # delta for E1
            lambda z: z[2]/z[1],        # iota/l for B1
            lambda z: z[2]              # iota for B0
        ]

        if self.PCE:
            self.basis = basis
            self.expansion = self.basis.generate_coefficients_multiple(self.fn)

        self.states = np.zeros([self.n, self.N + 1])
        self.states[:, 0] = x0

        if self.PCE:
            self.pce_coefs = np.zeros([self.basis.L, self.n, self.N + 1])
            self.pce_coefs[0, :, 0] = x0

        self.update_matrices(0)
        self.predict(0, self.N)
        

    def f(self, x, u):

        dt = self.dt
        delta, l, intent = self.param

        xx, yy, theta, v = x[0], x[1], x[2], x[3]
        gamma, a = (intent + self.intent_offset) * u * self.intent_gain
        xx += dt * v * math.cos(theta + gamma)
        yy += dt * v * math.sin(theta + gamma)
        theta += dt * v * math.sin(gamma)/l
        v += dt *  (a + delta)
        return np.array([xx, yy, theta, v])
    
    def g(self, x, u):
        return x
    
    def __get_linear_scalar(self):

        f0, f1, f2 = self.fn

        a0 = 1
        b0 = f2(self.param)
        e0 = 1
        a1 = 1
        b1 = f1(self.param)
        e1 = f0(self.param)

        return (a0, a1), (b0, b1), (e0, e1)
    

    def apply_control(self, t, u):

        self.states[:, t + 1] = self.f(self.states[:, t], u)
        
        if self.PCE:
            self.__predict_pce(t, t + 1)
            self.pce_coefs[0, :, t + 1] = self.f(self.pce_coefs[0, :, t], self.useq[:, t])
    
    def update_matrices(self, t):
        self.__update_lin_matrices(t)

        if self.PCE:
            self.__update_pce_matrices(t)

    def __update_lin_matrices(self, t):

        A, B, E = get_linear_matrix(self.states[:, t], self.dt)
        a, b, e = self.__get_linear_scalar()

        self.Al = sum([a[i] * A[i] for i in [0, 1]])
        self.Bl = sum([b[i] * B[i] for i in [0, 1]])
        self.Cl = np.zeros((self.m, self.n))
        self.Dl = np.zeros((self.m, self.m))
        self.El = sum([e[i] * E[i] for i in [0, 1]])


    def __update_pce_matrices(self, t):

        A, B, E = get_linear_matrix(self.states[:, t], self.dt)

        b_hat_0 = self.expansion[2]
        b_hat_1 = self.expansion[1]
        e_hat_1 = self.expansion[0]

        a_hat_0 = np.zeros(b_hat_1.shape)
        a_hat_1 = np.zeros(b_hat_1.shape)
        e_hat_0 = np.zeros(b_hat_1.shape)

        a_hat_0[0] = 1
        a_hat_1[0] = 1
        e_hat_0[0] = 1

        a_hat = np.array([a_hat_0, a_hat_1])
        b_hat = np.array([b_hat_0, b_hat_1])
        e_hat = np.array([e_hat_0, e_hat_1])
        
        self.Ap = np.array([[sum([a_hat[i] @ self.basis.psi[s][j] * A[i] for i in [0, 1]]) for j in range(self.basis.L)] for s in range(self.basis.L)])
        self.Bp = np.array([sum([b_hat[i][s] * B[i] for i in [0, 1]]) for s in range(self.basis.L)])
        self.Cp = np.zeros((self.m, self.basis.L * self.n))
        self.Dp = np.zeros((self.m, self.m))
        self.Ep = np.array([sum([e_hat[i][s] * E[i] for i in [0, 1]]) for s in range(self.basis.L)])

    def predict(self, t1, t2):

        assert self.useq.shape[1] >= t2

        if self.PCE:
            self.__predict_pce(t1, t2)
        self.__predict_non_lin(t1, t2)

    def __predict_non_lin(self, t1, t2):

        for t in range(t1, t2):
            self.states[:, t + 1] = self.f(self.states[:, t], self.useq[:, t])
    

    def __predict_lin(self, t1, t2):

        for t in range(t1, t2):
            self.states[:, t + 1] = self.states[:, t] + self.Al @ self.states[:, t] + self.Bl @ self.useq[:, t] + self.El

    
    def __predict_pce(self, t1, t2):

        for t in range(t1, t2):
            for s in range(self.basis.L):
                self.pce_coefs[s, :, t + 1] = self.pce_coefs[s, :, t] + sum([self.Ap[s][j] @ self.pce_coefs[j, :, t] for j in range(self.n)]) + self.Bp[s] @ self.useq[:, t] + self.Ep[s]
                for r in range(self.n):
                    if math.isnan(self.pce_coefs[s, r, t + 1]): 
                        self.pce_coefs[s, r, t + 1] = 0


class LinearAffineSystem(LinearSystem):
    """
    A linear affine discrete-time system of the form

    .. math::

        x_{t+1} = A x_t + B u_t + E

        y_t = C x_t + D u_t

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param A: A ``(n,n)`` numpy array representing the state transition matrix
    :param B: A ``(n,m)`` numpy array representing the control input matrix
    :param E: A ``(n, )`` numpy array representing the affine disturbance vector
    :param C: A ``(p,n)`` numpy array representing the state output matrix
    :param D: A ``(p,m)`` numpy array representing the control output matrix
    """
    def __init__(self, A, B, C, D, E):

        super().__init__(A, B, C, D)

        # Sanity checks on matrix sizes
        assert E.shape == (self.n, ), "E must be an (n,) vector"

        # Store dynamics parameters
        self.E = E

        # Dynamics functions
        self.dynamics_fcn = lambda x, u: A@x + B@u + E
        self.output_fcn = lambda x, u: C@x + D@u