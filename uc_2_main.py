import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from stlpy.systems.linear import DoubleIntegrator
from config.uc_2_config import visualize3D, visualize, target_specs, safety_specs
from config.uc_2_config import l as lane
import math
from libs.pce_basis import PCEBasis

import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)

Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle lengthS
q = 2                       # The polynomial order
N = 30                      # The control horizon

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)

gamma1 = np.linspace(0, 0, N)
a1 = np.linspace(0, -1.2, N)
u1 = np.array([gamma1, a1])

gamma2 = np.linspace(0, 0, N)
a2 = np.linspace(0, 0.5, N)
u2 = np.array([gamma2, a2])

# Generate the PCE instance and the specification
bias1 = cp.Normal(0, 1e-2)
length1 = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
intent1 = cp.DiscreteUniform(-1, 1)
eta1 = cp.J(bias1, length1, intent1) # Generate the random variable instance
B1 = PCEBasis(eta1, q)

bias2 = cp.Normal(0, 0.01)
length2 = cp.Uniform(lower=0.5-1e-3, upper=0.5+1e-3)
# intent_p = cp.Binomial(1, 0.3)
intent2 = cp.DiscreteUniform(0, 1)
eta2 = cp.J(bias2, length2, intent2) # Generate the random variable instance
B2 = PCEBasis(eta2, q)

phi_oppo = safety_specs(B1, N, "oppo")
phi_pedes = safety_specs(B2, N, "pedes")
phi_ego = target_specs(B1, N, "ego")

e0 = np.array([-lane*2, -lane/2, 0, 0.5])            # Initial position of the ego vehicle (EV)
o0 = np.array([86, lane/2, math.pi, 8])           # Initial position of the obstacle vehicle (OV)
p0 = np.array([1.2*lane, 1.2*lane, math.pi, 0])

ego = BicycleModel(e0, [0, l, 1], Ts, name="ego", color='red')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(o0, [0, l, 1], Ts, useq=u1, basis=B1, pce=True, name="oppo", color=(0, 0, 0.5))     # Dynamic model of the obstacle vehicle (OV)
pedes = BicycleModel(p0, [0, l, 1], Ts, useq=u2, basis=B2, pce=True, name="pedes", color=(1, 0.6, 0.2))

phi = phi_ego #& phi_oppo & phi_pedes
# Initialize the solver

sys = {ego.name: ego,
       oppo.name: oppo,
       pedes.name: pedes}


solver = PCEMICPSolver(phi, sys, N, robustness_cost=True)
R = np.array([[10, 0], [0, 500]])
solver.AddQuadraticControlCost(R)


# Solve the problem
x, u, _, _ = solver.Solve()

oz = solver.predict["oppo"]
pz = solver.predict["pedes"]
# model_checking(x, z, phi, 0)

# np.save('x.npy', x)

# x = np.load('x.npy')
# visualize3D(x, [oppo, pedes])
visualize(x, [oppo, pedes])
