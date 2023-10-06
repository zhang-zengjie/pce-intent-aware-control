import numpy as np
import chaospy as cp
from itertools import product
import numpoly

base_sampling_time = 0.1
base_length = 5
p = 3
np.random.seed(0)

length = cp.Trunc(cp.Normal(base_length, 0.05), lower=base_length - 0.05, upper=base_length + 0.05)
tau = cp.Trunc(cp.Normal(base_sampling_time, 0.01), lower=base_sampling_time - 0.01, upper=base_sampling_time + 0.01)
eta = cp.J(tau, length)

basis = cp.generate_expansion(order=p, dist=eta)
nodes, weights = cp.generate_quadrature(order=p, dist=eta, rule="Gaussian")
samples_a1 = [node[0] for node in nodes.T]
samples_a2 = [node[0]/node[1] for node in nodes.T]
_, a1_hat = cp.fit_quadrature(basis, nodes, weights, samples_a1, retall=True)
_, a2_hat = cp.fit_quadrature(basis, nodes, weights, samples_a2, retall=True)

L = len(basis)

Psi = np.zeros([L, L, L])

for s, j, i in product(range(L), range(L), range(L)):
    Psi[s][j][i] = cp.E(basis[s]*basis[j]*basis[i], eta)/cp.E(basis[s]*basis[s], eta)

np.save('psi.npy', Psi)
numpoly.save('basis.npy', basis)
np.save('a_hat.npy', np.array([a1_hat, a2_hat]))
