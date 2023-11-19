import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import gen_pce_specs, lanes, visualize
from libs.commons import model_checking
from libs.pce_basis import PCEBasis
import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)
ASSUMED_INPUT = "switch_lane"
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
intent = cp.Normal(1, 1e-3)
eta = cp.J(bias, length, intent) # Generate the random variable instance
B = PCEBasis(eta, q)

e0 = np.array([0, lanes['fast'], 0, 25])            # Initial position of the ego vehicle (EV)
o0 = np.array([50, lanes['slow'], 0, 25])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification
phi = gen_pce_specs(B, N, "oppo")

ego = BicycleModel(e0, [0, l, 1], Ts, name="ego")                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(o0, [0, l, 1], Ts, useq=v, basis=B, pce=True, name="oppo")     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

solver = PCEMICPSolver(phi, sys, N, robustness_cost=False)

# Adding input constraints (not necessary if input is in the cost function)
# u_min = np.array([[-0.5, -50]]).T
# u_max = np.array([[0.5, 50]]).T
# solver.AddControlBounds(u_min, u_max)

# Adding input to the cost function
Q = np.zeros([ego.n, ego.n])
R = np.array([[1e4, 0], [0, 1e-4]])
ref = np.array([0, lanes['fast'], 0, 0])
solver.AddQuadraticCost(Q, R, ref)

# Solve the problem
x, u, _, _ = solver.Solve()

z = solver.predict["oppo"]
# print(model_checking(x, z, phs, 0))

visualize(x, oppo)
