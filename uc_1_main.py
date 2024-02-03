import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import gen_pce_specs, lanes, visualize, select_intension
from libs.commons import model_checking
from libs.pce_basis import PCEBasis
import chaospy as cp
import math


Ts = 1    # The discrete sampling time Delta_t
l = 4       # The baseline value of the vehicle length
q = 2       # The polynomial order
N = 15      # The control horizon
M = 10
v0 = 10
sigma = 0.1
R = np.array([[1e4, 0], [0, 1e-4]])

mode = 2    # Select intention mode: 
            # 0 for switching-lane OV 
            # 1 for constant-speed OV
            # 2 for speeding-up OV

# The assumed control mode of the obstacle vehicle (OV)
v = select_intension(N, mode)

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

UPDATE_END = N-3

np.random.seed(7)

if True:
    
    xx = np.zeros([ego.n, N + 1, M])
    zz = np.zeros([oppo.n, N + 1, M])
    nodes = oppo.basis.eta.sample([N, M])

    for j in range(0, M):
        
        xx[:, 0, j] = e0
        zz[:, 0, j] = o0
        u_opt = np.zeros((2, ))

        for i in range(0, N):
            
            ego.x0 = xx[:, i, j]
            oppo.x0 = zz[:, i, j]
            oppo.param = np.array([nodes[0, i, j], nodes[1, i, j], 1])
            oppo.update_matrices()

            if i < UPDATE_END:
                phi = gen_pce_specs(B, N-i, v0*1.2, 12, 'oppo')
                solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
                solver.AddQuadraticCost(R)
                x, u, rho, _ = solver.Solve()
                if rho >= 0:
                    u_opt = u[:, 0]

            xx[:, i + 1, j] = ego.f(xx[:, i, j], u_opt)
            zz[:, i + 1, j] = oppo.f(zz[:, i, j], v[:, i])

        np.save('results/case_1/x_' + str(mode) + '_seed_' + str(j) + '_c.npy', xx[:, :, j])
        np.save('results/case_1/z_' + str(mode) + '_seed_' + str(j) + '_c.npy', zz[:, :, j])

else:

    xx = np.zeros([ego.n, N + 1, M])
    zz = np.zeros([oppo.n, N + 1, M])

    for j in range(0, M):
        xx[:, :, j] = np.load('results/case_1/x_' + str(mode) + '_seed_' + str(j) + '_c.npy')
        zz[:, :, j] = np.load('results/case_1/z_' + str(mode) + '_seed_' + str(j) + '_c.npy')

    visualize(xx[:, :N-3, :], zz[:, :N-3, :], mode)
