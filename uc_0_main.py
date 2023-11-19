import numpy as np
from matplotlib import pyplot
from libs.pce_basis import PCEBasis
import chaospy as cp
from libs.bicycle_model import BicycleModel


Ts = 0.5
l = 4
q = 2
N = 30

x0 = np.array([10, 0.5, 0, 25])

gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
u = np.array([gamma, a])

np.random.seed(7)

bias = cp.Trunc(cp.Normal(0, 0.01), lower=-0.01, upper=0.01)            # Random variable for control bias
length = cp.Uniform(lower=l - 1e-3, upper=l + 1e-3)                     # Random variable for car length
intent = cp.Normal(1, 1e-3)       # Random variable for intent uncertainty
eta = cp.J(bias, length, intent)

B = PCEBasis(eta, q)
M = 64

nodes = eta.sample([M,])

bicycle_pce = BicycleModel(x0, [0, l, 1], Ts, useq=u, basis=B, pce=True, name="pce")
bicycle_pce.update_initial(x0)
x_hat = bicycle_pce.predict_pce(N)

bicycle = BicycleModel(x0, [0, l, 1], Ts, useq=u, name="nonlinear")
mc_samples = np.zeros([M, 4, N + 1])
for i in range(M):
    bicycle.update_initial(x0)
    bicycle.update_parameter(nodes[:, i])
    mc_samples[i] = bicycle.predict(N)

bicycle_linear = BicycleModel(x0, [0, l, 1], Ts, useq=u, name="linear")
mc_samples_linear = np.zeros([M, 4, N + 1])
for i in range(M):
    bicycle_linear.update_initial(x0)
    bicycle_linear.update_parameter(nodes[:, i])
    mc_samples_linear[i] = bicycle_linear.predict_linear(N)
        

pce_mean = np.array([B.get_mean_from_coef(x_hat[:, :, i]) for i in range(N + 1)])
pce_var = np.array([B.get_var_from_coef(x_hat[:, :, i]) for i in range(N + 1)])
pce_std = np.array([B.get_std_from_coef(x_hat[:, :, i]) for i in range(N + 1)])
# pce_max = np.array([B.get_max_coef(x_hat[i]) for i in range(N + 1)])
# pce_coef = np.array([B.get_coef(x_hat[i], 0) for i in range(N + 1)])

# Draw plots: mean

pyplot.plot(np.linspace(0, 1, N+1), pce_mean.T[0])
pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples, 0)[0])
# pyplot.plot(np.linspace(0, 1, N+1), np.mean(mc_samples_linear, 0)[0])


# Draw plots: variance
'''
pyplot.plot(np.linspace(0, 1, N+1), pce_var.T[0])
# pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples, 0)[0])
pyplot.plot(np.linspace(0, 1, N+1), np.var(mc_samples_linear, 0)[0])
'''
pyplot.show()
