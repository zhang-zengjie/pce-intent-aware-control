import numpy as np
from uc_1_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from uc_1_config import gen_pce_specs, lanes, visualize, model_checking

import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)
ASSUMED_INPUT = "speed_up"
Ts = 0.5    # The discrete sampling time Delta_t
l = 4       # The baseline value of the vehicle length
q = 2       # The polynomial order
N = 30      # The control horizon

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
if ASSUMED_INPUT == "switch_lane":  # That the OV is trying to switch to the fast lane
    sigma = 0.1
    v = np.load('v.npy')
elif ASSUMED_INPUT == "slow_down":  # That the OV is trying to slow down (intention aware)
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, -2, N)
    v = np.array([gamma, a])
elif ASSUMED_INPUT == "speed_up":   # That the OV is trying to speed_up (adversarial action)
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 2, N)
    v = np.array([gamma, a])
else: # "big_variance"              # That the OV maintains the current speed with big variance (naive guess)
    sigma = 0.5
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 0, N)
    v = np.array([gamma, a])

bias = cp.Trunc(cp.Normal(0, sigma), lower=-sigma, upper=sigma)
length = cp.Uniform(lower=l - 1e-2, upper=l + 1e-2)
# intent = cp.DiscreteUniform(-1, 1)
intent = cp.Trunc(cp.Normal(0, 1e-5), lower=-1e-5, upper=1e-5)
eta = cp.J(bias, length, intent) # Generate the random variable instance

x0 = np.array([0, lanes['fast'], 0, 25])            # Initial position of the ego vehicle (EV)
z0 = np.array([50, lanes['slow'], 0, 25])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification
B, phi = gen_pce_specs(q, N, eta)

sys1 = BicycleModel(x0, [0, l, 1], Ts)                  # Dynamic model of the ego vehicle (EV)
sys2 = BicycleModel(z0, [0, l, 1], Ts, B, pce=True)     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
solver = PCEMICPSolver(phi, sys1, [sys2], [v], N, robustness_cost=False)

# Adding input constraints (not necessary if input is in the cost function)
# u_min = np.array([[-0.5, -50]]).T
# u_max = np.array([[0.5, 50]]).T
# solver.AddControlBounds(u_min, u_max)

# Adding input to the cost function
Q = np.zeros([sys1.n, sys1.n])
R = np.array([[1e4, 0], [0, 1e-4]])
ref = np.array([0, lanes['fast'], 0, 0])
solver.AddQuadraticCost(Q, R, ref)

# Solve the problem
x, z, u, _, _ = solver.Solve()

# model_checking(x, z, phi, 0)

visualize(x, z0, v, B, sys2)
