import numpy as np
import numpoly
from libs.commons import gen_pce_coefficients, monte_carlo_linear_bicycle, monte_carlo_bicycle
from libs.statistic import get_var_from_pce, get_mean_from_pce
from matplotlib import pyplot
from libs.pce_basis import PCEBasis
import chaospy as cp
from libs.bicycle_model import BicycleModel
from itertools import product
from libs.bicycle_model import get_linear_matrix


N = 30

base_sampling_time = 0.5
base_length = 4
q = 2

zeta_0 = np.array([10, 0.5, 0.02, 10])

gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
u = np.array([gamma, a]).T

np.random.seed(7)

length = cp.Trunc(cp.Normal(base_length, 0.1), lower=base_length - 0.1, upper=base_length + 0.1)
tau = cp.Trunc(cp.Normal(base_sampling_time, 0.05), lower=base_sampling_time - 0.05, upper=base_sampling_time + 0.05)
eta = cp.J(tau, length)

B = PCEBasis(eta, q)
M = 64

nodes = eta.sample([M, N])
bicycle = BicycleModel(zeta_0, nodes[:, 0, 0], B)

#f1 = lambda node: node[0]
#f2 = lambda node: node[0]/node[1]

# a1_hat = B.generate_coefficients(f1)
#a_hat = B.generate_coefficients_multiple([f1, f2])

# a_hat = np.array([a1_hat, a2_hat])

x_hat = np.zeros([N + 1, B.L, 4])
x_hat[0][0] = zeta_0
for i in range(N):
        x_hat[i + 1, :, :] = np.array([x_hat[i, s, :] + sum([bicycle.Ap[s][j] @ x_hat[i, j, :] for j in range(B.L)]) + bicycle.Bp[s] @ u[s] for s in range(B.L)])

# Propagate PCE
# zeta_hat = gen_pce_coefficients(N, zeta_0, u, B.psi, a_hat)


# Monte Carlo
'''
M = 64
nodes = eta.sample([M, N + 1])

mc_samples = np.zeros([M, N + 1, 4])

mc_samples[:, 0, :] = zeta_0

bicycle = BicycleModel(zeta_0, nodes[:, 0, 0])
for i in range(M):
    for j in range(N):
        bicycle.update_parameter(nodes[:, i, j])
        mc_samples[i, j+1, :] = bicycle.f(mc_samples[i, j, :], u[j])
'''




mc_samples = np.zeros([M, N + 1, 4])
mc_samples[:, 0, :] = zeta_0
for i, j in product(range(M), range(N)):
        bicycle.update_parameter(nodes[:, i, j])
        mc_samples[i, j+1, :] = bicycle.f(mc_samples[i, j, :], u[j])


# mc_samples = np.array([monte_carlo_bicycle(N, zeta_0, u, node[0], node[1]) for node in nodes.T])
# mc_samples_linear = np.array([monte_carlo_linear_bicycle(N, zeta_0, u, node[0], node[1]) for node in nodes.T])


mc_samples_linear = np.zeros([M, N + 1, 4])
mc_samples_linear[:, 0, :] = zeta_0
for i, j in product(range(M), range(N)):
        bicycle.update_parameter(nodes[:, i, j])
        mc_samples_linear[i, j+1, :] = mc_samples_linear[i, j, :] + bicycle.sys.A @ mc_samples_linear[i, j, :] + bicycle.sys.B @ u[j]


# Draw plots: mean
'''
pyplot.plot(np.linspace(0, 1, N+1), B.get_mean_from_coef(x_hat).T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples, 0).T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples_linear, 0).T[0])

# Draw plots: variance
'''
# pyplot.plot(np.linspace(0, 1, N+1), B.get_var_from_coef(x_hat).T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples, 0).T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples_linear, 0).T[1])


pyplot.show()