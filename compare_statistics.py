import numpy as np
from matplotlib import pyplot
from libs.pce_basis import PCEBasis
import chaospy as cp
from libs.bicycle_model import BicycleModel


N = 30

base_sampling_time = 0.5
base_length = 4
q = 2

zeta_0 = np.array([10, 0.5, 0.02, 10])

gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
u = np.array([gamma, a])

np.random.seed(7)

length = cp.Trunc(cp.Normal(base_length, 0.1), lower=base_length - 0.1, upper=base_length + 0.1)
tau = cp.Trunc(cp.Normal(base_sampling_time, 0.05), lower=base_sampling_time - 0.05, upper=base_sampling_time + 0.05)
eta = cp.J(tau, length)

B = PCEBasis(eta, q)
M = 64

nodes = eta.sample([M,])
bicycle_pce = BicycleModel(zeta_0, nodes[:, 0], B, pce=True)


x_hat = np.zeros([N + 1, B.L, 4])
x_hat[0][0] = zeta_0
for i in range(N):
        x_hat[i + 1, :, :] = np.array([x_hat[i, s, :] + sum([bicycle_pce.Ap[s][j] @ x_hat[i, j, :] for j in range(B.L)]) + bicycle_pce.Bp[s] @ u[:, s] for s in range(B.L)]) + bicycle_pce.Ep


bicycle = BicycleModel(zeta_0, nodes[:, 0])

mc_samples = np.zeros([M, N + 1, 4])
mc_samples[:, 0, :] = zeta_0
for i in range(M):
    bicycle.update_parameter(nodes[:, i])
    for j in range(N):
        mc_samples[i, j+1, :] = bicycle.f(mc_samples[i, j, :], u[:, j])

mc_samples_linear = np.zeros([M, N + 1, 4])
mc_samples_linear[:, 0, :] = zeta_0
for i in range(M):
    bicycle.update_parameter(nodes[:, i])
    for j in range(N):
        mc_samples_linear[i, j + 1, :] = mc_samples_linear[i, j, :] + bicycle.Al @ mc_samples_linear[i, j, :] + bicycle.Bl @ u[:, j] + bicycle.El


pce_mean = np.array([B.get_mean_from_coef(x_hat[i]) for i in range(N + 1)])
pce_var = np.array([B.get_var_from_coef(x_hat[i]) for i in range(N + 1)])

# Draw plots: mean

pyplot.plot(np.linspace(0, 1, N+1), pce_mean.T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples, 0).T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples_linear, 0).T[0])

# Draw plots: variance
'''
pyplot.plot(np.linspace(0, 1, N+1), pce_var.T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples, 0).T[1])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples_linear, 0).T[1])
'''

pyplot.show()