import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from commons import gen_pce_coefficients, monte_carlo_linear_bicycle, monte_carlo_bicycle
from gen_basis import eta
from param import lanes, N, v


u = np.load('u.npy')
x = np.load('x.npy')
z = np.load('z.npy')

H = 10

plt.plot(lanes['left'] * np.ones((H, )))
plt.plot(lanes['middle'] * np.ones((H, )))
plt.plot(lanes['right'] * np.ones((H, )))

p1, = plt.plot(x[0, :], x[1, :])

# p2, = plt.plot(z[0, 0, :], z[0, 1, :])

M = 64
nodes = eta.sample([M, ])
mc_samples = np.array([monte_carlo_bicycle(N, z[0, :, 0], v.T, node[0], node[1]) for node in nodes.T])

for i in range(M):
    plt.plot(mc_samples[i, :, 0], mc_samples[i, :, 1])

plt.xlim([0, H])
# plt.ylim([0, 5])
plt.xlabel('x')
plt.ylabel('y')
# plt.legend([p1, p2], ['ego', 'obstacle'], loc='lower left')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.show()
