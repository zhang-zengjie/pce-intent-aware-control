import numpy as np
from libs.pce_micp_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from param import gen_pce_specs, lanes
from plot_results import visualize


base_sampling_time = 0.5
base_length = 4
q = 2
N = 30

B, phi = gen_pce_specs(base_sampling_time, base_length, q, N)

gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
v = np.array([gamma, a])

x0 = np.array([0, lanes['fast'], 0, 25])
z0 = np.array([50, lanes['slow'], 0, 25])

sys1 = BicycleModel(x0, [base_sampling_time, base_length])
sys2 = BicycleModel(z0, [base_sampling_time, base_length], B, pce=True)

solver = PCEMICPSolver(phi, sys1, sys2, v, N, robustness_cost=True)

u_min = np.array([[-0.5, 0]]).T
u_max = np.array([[0.5, 10]]).T

Q = np.zeros([sys1.n, sys1.n])
R = np.array([[0.5, 0], [0, 0.01]])
solver.AddQuadraticCost(Q, R)
#solver.AddControlBounds(u_min, u_max)
x, z, u, _, _ = solver.Solve()
#np.save('x.npy', x)
#np.save('z.npy', z)
#np.save('u.npy', u)

visualize(x, z0, v, B, sys2)