import numpy as np
from commons import gen_linear_matrix, pce_model, bicycle_model


a_hat = np.load('a_hat.npy')
psi = np.load('psi.npy')
L = a_hat.shape[1]

N = 10
x0 = 0
y0 = 0
theta0 = 0
v0 = 5
gamma0 = 0.1

gamma = np.linspace(gamma0, 0, N)
a = np.linspace(0, 0, N)

zeta_hat = np.zeros([L, 4])
xi_0 = np.array([x0, y0, theta0, v0])
zeta_hat[0] = xi_0
u_0 = np.array([gamma[0], a[0]])

for i in range(N):
    mu = np.array([gamma[i], a[i]])
    zeta_hat = pce_model(zeta_hat, mu, psi, xi_0, u_0, a_hat)

