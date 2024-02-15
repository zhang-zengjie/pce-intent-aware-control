import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import *
from libs.commons import model_checking


Ts = 1    # The discrete sampling time Delta_t
l = 4       # The baseline value of the vehicle length
N = 15      # The control horizon
M = 100
R = np.array([[1e4, 0], [0, 1e-2]])

mode = 0    # Select intention mode: 
            # 0 for switching-lane OV 
            # 1 for constant-speed OV
            # 2 for speeding-up OV

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
v = get_intension(N, mode)
B = gen_bases(l)

v0 = 10
e0 = np.array([0, lanes['fast'], 0, v0*1.33])            # Initial position of the ego vehicle (EV)
o0 = np.array([2*v0, lanes['slow'], 0, v0])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification

ego = BicycleModel(Ts, name='ego')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(Ts, useq=v, basis=B, pce=True, name='oppo')     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

xx = np.zeros([ego.n, N + 1])
zz = np.zeros([oppo.basis.L, oppo.n, N + 1])

if False:
        
    xx[:, 0,] = e0
    zz[0, :, 0] = o0
    u_opt = np.zeros((2, ))

    ego.update_param(np.array([0, l, 1]))
    oppo.update_param(np.array([0, l, 1]))

    for i in range(0, N):
        
        # Update specification
        phi = gen_pce_specs(B, N-i, v0*1.2, 'oppo')

        # Update current states and parameters
        ego.update_initial(xx[:, i])
        oppo.update_initial(zz[0, :, i])
        oppo.update_initial_pce(zz[:, :, i])

        ego.update_matrices()
        oppo.update_matrices()

        # Solve
        solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
        solver.AddQuadraticCost(R)
        x, u, rho, _ = solver.Solve()

        # In case infeasibility
        if rho >= 0:
            u_opt = u[:, 0]

        # Probabilistic prediction
        zz[:, :, i + 1] = oppo.predict_pce(1)[:, :, 1]

        # Simulate the next step
        xx[:, i + 1] = ego.f(xx[:, i], u_opt)
        zz[0, :, i + 1] = oppo.f(zz[0, :, i], v[:, i])

    np.save('results/case_1/x_mode_' + str(mode) + '.npy', xx)
    np.save('results/case_1/z_mode_' + str(mode) + '.npy', zz)

else:

    xx = np.load('results/case_1/x_mode_' + str(mode) + '.npy')

zz_s = np.zeros([oppo.n, N + 1, M])
samples = oppo.basis.eta.sample([M, ])
for j in range(0, M):
    oppo.update_param(np.array([samples[0, j], samples[1, j], 1]))
    oppo.update_initial(o0)
    zz_s[:, :, j] = oppo.predict(N)

visualize(xx[:, :N-1], zz_s[:, :, :N-1], mode)
