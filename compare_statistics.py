import numpy as np
import numpoly
from commons import eta, gen_pce_coefficients, monte_carlo_linear_bicycle, monte_carlo_bicycle
from statistics import get_var_from_pce, get_mean_from_pce
from matplotlib import pyplot


a_hat = np.load('a_hat.npy')
psi = np.load('psi.npy')
basis = numpoly.load('basis.npy')
L = a_hat.shape[1]

N = 10
x0 = 0
y0 = 0
theta0 = 0
v0 = 10
zeta_0 = np.array([x0, y0, theta0, v0])

gamma = np.linspace(0.01, 0, N)
a = np.linspace(0, 0, N)
u = np.array([gamma, a]).T

# Propagate PCE
zeta_hat = gen_pce_coefficients(N, zeta_0, u, psi, a_hat)


# Monte Carlo
M = 64
nodes = eta.sample([M, ])
mc_samples = np.array([monte_carlo_bicycle(N, zeta_0, u, node[0], node[1]) for node in nodes.T])
mc_samples_linear = np.array([monte_carlo_linear_bicycle(N, zeta_0, u, node[0], node[1]) for node in nodes.T])


# Draw plots: mean

pyplot.plot(np.linspace(0, 1, N+1), get_mean_from_pce(zeta_hat).T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples, 0).T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples_linear, 0).T[1])


# Draw plots: variance
'''
pyplot.plot(np.linspace(0, 1, N+1), get_var_from_pce(zeta_hat, basis, eta).T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples, 0).T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples_linear, 0).T[0])
'''
pyplot.show()
