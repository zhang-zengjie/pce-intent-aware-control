import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_2_config import visualize, turn_specs, safety_specs, gen_bases, get_intentions
from config.uc_2_config import l as lane
import math


RECAL = True

Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle length
N = 30                      # The control horizon
M = 1
R = np.array([[10, 0], [0, 500]])

mode = 2    # Select simulation mode: 
            # 0 for no_reaction 
            # 1 for reaction with proposed method
            # 2 for reaction with conventional method

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
v1, v2 = get_intentions(N)
B1, B2 = gen_bases(mode)

e0 = np.array([-lane*2, -lane/2, 0, 0.5])            # Initial position of the ego vehicle (EV)
o0 = np.array([95, lane/2, math.pi, 8])           # Initial position of the obstacle vehicle (OV)
p0 = np.array([1.2*lane, 1.2*lane, math.pi, 0])

ego = BicycleModel(Ts, name="ego", color='red')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(Ts, useq=v1, basis=B1, pce=True, name="oppo", color=(0, 0, 0.5))     # Dynamic model of the obstacle vehicle (OV)
pedes = BicycleModel(Ts, useq=v2, basis=B2, pce=True, name="pedes", color=(1, 0.6, 0.2))

sys = {ego.name: ego,
       oppo.name: oppo,
       pedes.name: pedes}

xx_ego = np.zeros([ego.n, N + 1, M])
xx_oppo = np.zeros([oppo.n, N + 1, M])
xx_pedes = np.zeros([pedes.n, N + 1, M])
j = 0

if RECAL:

    xx_ego[:, 0, j] = e0
    xx_oppo[:, 0, j] = o0
    xx_pedes[:, 0, j] = p0

    nodes_o = oppo.basis.eta.sample([N, M])
    nodes_p = pedes.basis.eta.sample([N, M])

    u_opt = np.zeros((2, ))

    for i in range(0, N):
        
        ego.x0 = xx_ego[:, i, j]
        oppo.x0 = xx_oppo[:, i, j]
        pedes.x0 = xx_pedes[:, i, j]

        ego.param = np.array([0, l, 1])
        oppo.param = np.array(nodes_o[:, i, j])
        pedes.param = np.array(nodes_p[:, i, j])

        ego.update_matrices()
        oppo.update_matrices()
        pedes.update_matrices()

        phi_oppo = safety_specs(B1, N-i, "oppo")
        phi_pedes = safety_specs(B2, N-i, "pedes")
        phi_ego = turn_specs(B1, N-i, "ego")

        if mode == 0:
            phi = phi_ego
        else:
            phi = phi_ego & phi_oppo & phi_pedes

        solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
        solver.AddQuadraticCost(R)
        x, u, rho, _ = solver.Solve()
        
        if rho >= 0:
            u_opt = u[:, 0]
            
        xx_ego[:, i + 1, j] = ego.f(xx_ego[:, i, j], u_opt)
        xx_oppo[:, i + 1, j] = oppo.f(xx_oppo[:, i, j], v1[:, i])
        xx_pedes[:, i + 1, j] = pedes.f(xx_pedes[:, i, j], v2[:, i])
        
    np.save('results/case_2/xx_ego_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', xx_ego[:, :, j])
    np.save('results/case_2/xx_oppo_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', xx_oppo[:, :, j])
    np.save('results/case_2/xx_pedes_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', xx_pedes[:, :, j])
    
else:

    xx_ego[:, :, j] = np.load('results/case_2/xx_ego_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')
    xx_oppo[:, :, j] = np.load('results/case_2/xx_oppo_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')
    xx_pedes[:, :, j] = np.load('results/case_2/xx_pedes_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')

oppo.x0 = o0
pedes.x0 = p0
visualize(xx_ego[:, :, j], [oppo, pedes], cursor=24)
