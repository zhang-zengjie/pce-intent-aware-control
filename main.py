import numpy as np
import numpoly
from commons import gen_pce_coefficients, monte_carlo_linear_bicycle, monte_carlo_bicycle
from statistic import get_var_from_pce, get_mean_from_pce
from matplotlib import pyplot
from pce_micp_solver import PCEMICPSolver
from param import N, phi, lanes, v
from commons import gen_bicycle_linear_sys
from gen_basis import base_length, base_sampling_time

x0 = np.array([0, lanes['fast'], 0, 1.2])
z0 = np.array([1, lanes['slow'], 0, 1.2])

sys = gen_bicycle_linear_sys(x0, base_sampling_time, base_length)

solver = PCEMICPSolver(phi, sys, x0, z0, v, N, robustness_cost=False)

Q = np.zeros([sys.n, sys.n])
R = 0.01 * np.eye(sys.m)
solver.AddQuadraticCost(Q, R)
x, z, u, _, _ = solver.Solve()
np.save('x.npy', x)
np.save('z.npy', z)
np.save('u.npy', u)