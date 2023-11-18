import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from stlpy.systems.linear import DoubleIntegrator
from config.uc_2_config import visualize, oppo_specs, pedes_specs
from config.uc_2_config import l as lane
import math

import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)

Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle length
q = 2                       # The polynomial order
N = 30                      # The control horizon

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)

gamma_o = np.linspace(0, 0, N)
a_o = np.linspace(0, -0.8, N)
ou = np.array([gamma_o, a_o])

gamma_h = np.linspace(0, 0, N)
a_h = np.linspace(0, 0.4, N)
hu = np.array([gamma_h, a_h])

# Generate the PCE instance and the specification

bias_o = cp.Normal(0, 1e-2)
length_o = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
intent_o = cp.DiscreteUniform(-1, 1)
eta_o = cp.J(bias_o, length_o, intent_o) # Generate the random variable instance
Bo, phi_o = oppo_specs(q, N, eta_o, "oppo")

bias_p = cp.Normal(0, 0.01)
length_p = cp.Uniform(lower=0.5-1e-3, upper=0.5+1e-3)
# intent_p = cp.Binomial(1, 0.3)
intent_p = cp.DiscreteUniform(0, 1)
eta_p = cp.J(bias_p, length_p, intent_p) # Generate the random variable instance
Bp, phi_p = pedes_specs(q, N, eta_p, "pedes")

x0 = np.array([-lane*2, -lane/2, 0, 0.5])            # Initial position of the ego vehicle (EV)
z0 = np.array([8*lane, lane/2, math.pi, 6])           # Initial position of the obstacle vehicle (OV)
h0 = np.array([1.2*lane, 1.2*lane, math.pi, 0])

ego = BicycleModel(x0, [0, l, 1], Ts, name="ego", color='red')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(z0, [0, l, 1], Ts, useq=ou, basis=Bo, pce=True, name="oppo", color=(0, 0, 0.5))     # Dynamic model of the obstacle vehicle (OV)
pedes = BicycleModel(h0, [0, l, 1], Ts, useq=hu, basis=Bp, pce=True, name="pedes", color=(1, 0.6, 0.2))

phi = phi_o & phi_p
# Initialize the solver
solver = PCEMICPSolver(phi, ego, [oppo, pedes], N, robustness_cost=True)

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

# visualize(x, z0, ou, Bo, Bp, oppo, pedes)
visualize(x, [oppo, pedes])