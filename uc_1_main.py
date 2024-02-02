import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import gen_pce_specs, lanes, visualize
from libs.commons import model_checking
from libs.pce_basis import PCEBasis
import chaospy as cp
import math


# The assumed control input of the obstacle vehicle (OV)
ASSUMED_INPUT = 'speed_up'
Ts = 1    # The discrete sampling time Delta_t
l = 4       # The baseline value of the vehicle length
q = 2       # The polynomial order
N = 15      # The control horizon
v0 = 10

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
if ASSUMED_INPUT == 'switch_lane':  # That the OV is trying to switch to the fast lane
    sigma = 0.1
    gamma = np.array([0.005 * math.sin(i*6.28/(N-1)) for i in range(N)])
    a = np.zeros([N, ])
    v = np.array([gamma, a])
elif ASSUMED_INPUT == 'constant_speed':  # That the OV is trying to slow down (intention aware)
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 0, N)
    v = np.array([gamma, a])
elif ASSUMED_INPUT == 'speed_up':   # That the OV is trying to speed_up (adversarial action)
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 2, N)
    v = np.array([gamma, a])
else: # 'big_variance'              # That the OV maintains the current speed with big variance (naive guess)
    sigma = 0.5
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 0, N)
    v = np.array([gamma, a])

bias = cp.Trunc(cp.Normal(0, sigma), lower=-sigma, upper=sigma)
length = cp.Uniform(lower=l - 1e-2, upper=l + 1e-2)
intent = cp.Normal(1, 1e-3)
eta = cp.J(bias, length, intent) # Generate the random variable instance
B = PCEBasis(eta, q)

e0 = np.array([0, lanes['fast'], 0, v0*1.33])            # Initial position of the ego vehicle (EV)
o0 = np.array([2*v0, lanes['slow'], 0, v0])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification

ego = BicycleModel(e0, [0, l, 1], Ts, name='ego')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(o0, [0, l, 1], Ts, useq=v, basis=B, pce=True, name='oppo')     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

nodes = oppo.basis.eta.sample([N, ])

UPDATE_END = N-3

if True:

    xx = np.zeros([ego.n, N + 1])
    zz = np.zeros([oppo.n, N + 1])
    xx[:, 0] = ego.x0
    zz[:, 0] = oppo.x0
    u_opt = np.zeros((2, ))

    for i in range(0, N):
        
        ego.x0 = xx[:, i]
        oppo.x0 = zz[:, i]
        oppo.param = np.array([nodes[0, i], nodes[1, i], 1])
        oppo.update_matrices()

        if i < UPDATE_END:
            phi = gen_pce_specs(B, N-i, v0*1.2, 20, 'oppo')
            solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
            x, u, rho, _ = solver.Solve()
            if rho >= 0:
                u_opt = u[:, 0]

        xx[:, i + 1] = ego.f(xx[:, i], u_opt)
        zz[:, i + 1] = oppo.f(zz[:, i], v[:, i])

    np.save('x_' + ASSUMED_INPUT + '_c.npy', xx)
    np.save('z_' + ASSUMED_INPUT + '_c.npy', zz)

else:
    xx = np.load('x_' + ASSUMED_INPUT + '_c.npy')
    zz = np.load('z_' + ASSUMED_INPUT + '_c.npy')

visualize(xx, zz, N-3, mode=ASSUMED_INPUT)
