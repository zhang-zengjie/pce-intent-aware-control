from stlpy.systems import NonlinearSystem, LinearSystem
import numpy as np
import math


def get_linear_matrix(x0):
    theta0, v0 = x0[2], x0[3]
    gamma0 = 0
    
    A1 = [[0, 0, 0, math.cos(theta0 + gamma0)],
          [0, 0, 0, math.sin(theta0 + gamma0)],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]
    '''
    A1 = [[0, 0, - v0 * math.sin(theta0 + gamma0), math.cos(theta0 + gamma0)],
          [0, 0, v0 * math.cos(theta0 + gamma0), math.sin(theta0 + gamma0)],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]
    '''

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

    def __init__(self, x0, param):


        self.n = 4
        self.m = 2
        self.p = 4
        self.x0 = None
        self.param = None

        self.update_initial(x0)
        self.update_parameter(param)

        self.fn = [
            lambda z: z[0],
            lambda z: z[0]/z[1]
        ]

        A, B, E = get_linear_matrix(x0)
        a, b, e = self.get_linear_scalar()

        Am = sum([a[i] * A[i] for i in [0, 1]])
        Bm = sum([b[i] * B[i] for i in [0, 1]])

        Cm = np.zeros((self.m, self.n))
        Dm = np.zeros((self.m, self.m))

        self.Em = e * E
        
        self.sys = LinearSystem(Am, Bm, Cm, Dm)

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
        