import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import gen_pce_specs, lanes, visualize
from libs.commons import model_checking
from libs.pce_basis import PCEBasis
import chaospy as cp
import math


# The assumed control input of the obstacle vehicle (OV)
ASSUMED_INPUT = "switch_lane"
Ts = 1    # The discrete sampling time Delta_t
l = 4       # The baseline value of the vehicle length
q = 2       # The polynomial order
N = 15      # The control horizon
v0 = 10

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
if ASSUMED_INPUT == "switch_lane":  # That the OV is trying to switch to the fast lane
    sigma = 0.1
    gamma = np.array([0.0005 * v0 * math.sin(i*6.28/(N-1)) for i in range(N)])
    a = np.zeros([N, ])
    v = np.array([gamma, a])
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

e0 = np.array([0, lanes['fast'], 0, v0*1.333])            # Initial position of the ego vehicle (EV)
o0 = np.array([2*v0, lanes['slow'], 0, v0])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification

ego = BicycleModel(e0, [0, l, 1], Ts, name="ego")                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(o0, [0, l, 1], Ts, useq=v, basis=B, pce=True, name="oppo")     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

nodes = oppo.basis.eta.sample([N, ])

UPDATE_END = N-3

if False:

    xx = np.zeros([ego.n, N + 1])
    zz = np.zeros([oppo.n, N + 1])
    xx[:, 0] = ego.x0
    zz[:, 0] = oppo.x0

    for i in range(0, UPDATE_END):
        
        ego.x0 = xx[:, i]
        oppo.x0 = zz[:, i]
        oppo.param = np.array([nodes[0, i], nodes[1, i], 1])
        oppo.update_matrices()

        
        phi = gen_pce_specs(B, N-i, v0*1.2, "oppo")
        solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
        x, u, _, _ = solver.Solve()

        xx[:, i + 1] = ego.f(xx[:, i], u[:, 0])
        zz[:, i + 1] = oppo.f(zz[:, i], v[:, i])

    for i in range(UPDATE_END, N):

        ego.x0 = xx[:, i]
        oppo.x0 = zz[:, i]
        oppo.param = np.array([nodes[0, i], nodes[1, i], 1])
        oppo.update_matrices()

        xx[:, i + 1] = ego.f(xx[:, i], u[:, 0])
        zz[:, i + 1] = oppo.f(zz[:, i], v[:, i])

    # print(model_checking(x, z, phs, 0))
    np.save('x_' + ASSUMED_INPUT + '_c.npy', xx)
    np.save('z_' + ASSUMED_INPUT + '_c.npy', zz)

else:
    xx = np.load('x_' + ASSUMED_INPUT + '_c.npy')
    zz = np.load('z_' + ASSUMED_INPUT + '_c.npy')
    
oppo.x0 = o0
visualize(xx, zz, x_range=[-5, 165], y_range=[0, 10], t_end=N)
