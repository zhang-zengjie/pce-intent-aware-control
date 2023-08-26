import numpy as np
import chaospy as cp
from itertools import product
from commons import eta, bicycle_model, bicycle_linear_model
from matplotlib import pyplot

N = 10


def monte_carlo_bicycle(state_0, input_0, control, delta_t, length):

    samples = np.zeros([N + 1, 4])
    samples[0] = state_0
    for k in range(N):
        samples[k+1] = bicycle_model(samples[k], control[k], delta_t, length)
    return samples


def monte_carlo_linear_bicycle(state_0, input_0, control, delta_t, length):

    samples = np.zeros([N + 1, 4])
    samples[0] = state_0
    for k in range(N):
        samples[k+1] = bicycle_linear_model(samples[k], control[k], state_0, input_0, delta_t, length)
    return samples


M = 64

nodes = eta.sample([M, ])

x0 = 0
y0 = 0
theta0 = 0
v0 = 10
zeta_0 = np.array([x0, y0, theta0, v0])

gamma = np.linspace(0.01, 0, N)
a = np.linspace(0, 0, N)
u = np.array([gamma, a]).T

mc_samples = np.array([monte_carlo_bicycle(zeta_0, u[0], u, node[0], node[1]) for node in nodes.T])
mc_samples_linear = np.array([monte_carlo_linear_bicycle(zeta_0, u[0], u, node[0], node[1]) for node in nodes.T])

for j in range(M):
    pyplot.plot(np.linspace(0, 1, N+1), mc_samples[j].T[1])

#for j in range(M):
#    pyplot.plot(np.linspace(0, 1, N+1), mc_samples_linear[j].T[1])

pyplot.show()
