import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from stlpy.systems.linear import DoubleIntegrator
from config.uc_2_config import visualize, gen_pce_specs
from config.uc_2_config import l as lane
import math

import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)
ASSUMED_INPUT = "speed_up"
Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle length
q = 2                       # The polynomial order
N = 30                      # The control horizon

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)

sigma = 0.01
gamma = np.linspace(0, 0, N)
a = np.linspace(0, -0.5, N)
ou = np.array([gamma, a])
hu = np.array([gamma, a])

bias = cp.Normal(0, sigma)
length = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
intent = cp.DiscreteUniform(-1, 1)
eta = cp.J(bias, length, intent) # Generate the random variable instance

x0 = np.array([-lane*2, -lane/2, 0, 0.5])            # Initial position of the ego vehicle (EV)
z0 = np.array([lane*3, lane/2, math.pi, 2])           # Initial position of the obstacle vehicle (OV)
h0 = np.array([lane, 1.2, lane, math.pi, 0.4])

# Generate the PCE instance and the specification
B, phi = gen_pce_specs(q, N, eta)

ego = BicycleModel(x0, [0, l, 1], Ts, name="ego")                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(z0, [0, l, 1], Ts, B, pce=True, name="oppo")     # Dynamic model of the obstacle vehicle (OV)

pedes = BicycleModel(z0, [0, l, 1], Ts, B, pce=True, name="pedes")

# Initialize the solver
solver = PCEMICPSolver(phi, ego, [oppo, pedes], [ou, hu], N, robustness_cost=False)

# Adding input constraints (not necessary if input is in the cost function)
# u_min = np.array([[-0.5, -50]]).T
# u_max = np.array([[0.5, 50]]).T
# solver.AddControlBounds(u_min, u_max)

# Adding input to the cost function
Q = np.zeros([ego.n, ego.n])
R = np.array([[1, 0], [0, 500]])
solver.AddQuadraticCost(Q, R)


# Solve the problem
x, u, _, _ = solver.Solve()

z = solver.predict[solver.index["oppo"]]

# model_checking(x, z, phi, 0)

visualize(x, z0, ou, B, oppo)
