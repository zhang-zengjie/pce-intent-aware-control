from stlpy.systems import NonlinearSystem, LinearSystem
import numpy as np
import math


def get_linear_matrix(x0):
    theta0, v0 = x0[2], x0[3]
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
    
    E = [v0 * math.sin(theta0 + gamma0) * theta0, - v0 * math.cos(theta0 + gamma0) * theta0, 0, 0]

    return np.array([A1, A2]), np.array([B1, B2]), np.array(E)


class BicycleModel(NonlinearSystem):

    def __init__(self, x0, param, basis=None, pce=False):

        self.n = 4
        self.m = 2
        self.p = 4

        self.basis = basis

        self.fn = [
            lambda z: z[0],
            lambda z: z[0]/z[1]
        ]

        self.update_initial(x0)
        self.update_parameter(param)

        if pce:
            self.update_pce_parameter()


    def f(self, x, u):

        delta_t = self.param[0]
        l = self.param[1]

        xx, yy, theta, v = x[0], x[1], x[2], x[3]
        gamma, a = u[0], u[1]
        xx += delta_t * v * math.cos(theta + gamma)
        yy += delta_t * v * math.sin(theta + gamma)
        theta += delta_t * v * math.sin(gamma)/l
        v += delta_t * a
        return np.array([xx, yy, theta, v])
    
    def g(self, x, u):
        return x
    
    def get_linear_scalar(self):

        f1, f2 = self.fn[0], self.fn[1]

        a1 = f1(self.param)
        b1 = f1(self.param)
        e = f1(self.param)
        a2 = f2(self.param)
        b2 = f2(self.param)

        return (a1, a2), (b1, b2), e
    
    def update_initial(self, x0):
        self.x0 = x0

    def update_parameter(self, param):

        self.param = param

        A, B, E = get_linear_matrix(self.x0)
        a, b, e = self.get_linear_scalar()

        self.Al = sum([a[i] * A[i] for i in [0, 1]])
        self.Bl = sum([b[i] * B[i] for i in [0, 1]])
        self.Cl = np.zeros((self.m, self.n))
        self.Dl = np.zeros((self.m, self.m))
        self.El = e * E


    def update_pce_parameter(self):

        A, B, E = get_linear_matrix(self.x0)

        a_hat = self.basis.generate_coefficients_multiple(self.fn)
        b_hat = a_hat
        e_hat = a_hat[0]

        self.Ap = np.array([[sum([a_hat[i] @ self.basis.psi[s][j] * A[i] for i in [0, 1]]) for j in range(self.basis.L)] for s in range(self.basis.L)])
        self.Bp = np.array([sum([b_hat[i][s] * B[i] for i in [0, 1]]) for s in range(self.basis.L)])
        self.Cp = np.zeros((self.m, self.basis.L * self.n))
        self.Dp = np.zeros((self.m, self.m))
        self.Ep = np.array([e_hat[s] * E for s in range(self.basis.L)])


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
    :param E: A ``(n,)`` numpy array representing the affine disturbance vector
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