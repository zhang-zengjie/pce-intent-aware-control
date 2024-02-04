import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_2_config import *


Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle length
N = 30                      # The control horizon
M = 1
R = np.array([[10, 0], [0, 500]])
Q = 25

mode = 1    # Select simulation mode: 
            # 0 for no_reaction 
            # 1 for reaction with proposed method
            # 2 for reaction with conventional method

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)
v1, v2 = get_intentions(N)
B1, B2 = gen_bases()
e0, o0, p0 = get_initials()

sys = {'ego': BicycleModel(Ts, name='ego'),                                     # Dynamic model of the ego vehicle (EV)
       'oppo': BicycleModel(Ts, useq=v1, basis=B1, pce=True, name='oppo'),      # Dynamic model of the obstacle vehicle (OV)
       'pedes': BicycleModel(Ts, useq=v2, basis=B2, pce=True, name='pedes')}    # Dynamic model of the pedestrian (PD)

tr = {'ego': np.zeros([sys['ego'].n, N + 1, M]),
      'oppo': np.zeros([sys['oppo'].n, N + 1, M]),
      'pedes': np.zeros([sys['pedes'].n, N + 1, M])}

samples = {'oppo_simulate': sys['oppo'].basis.eta.sample([N, M]),
           'pedes_simulate': sys['pedes'].basis.eta.sample([N, M]),
           'oppo_scenario': sys['oppo'].basis.eta.sample([N, Q]),
           'pedes_scenario': sys['pedes'].basis.eta.sample([N, Q])}

if True:
    '''
    nodes_o = sys['oppo'].basis.eta.sample([N, M])
    nodes_p = sys['pedes'].basis.eta.sample([N, M])

    if mode == 2:
        scena_o = sys['oppo'].basis.eta.sample([N, Q])
        scena_p = sys['pedes'].basis.eta.sample([N, Q])
    '''
    for j in range(0, M):

        tr['ego'][:, 0, j] = e0
        tr['oppo'][:, 0, j] = o0
        tr['pedes'][:, 0, j] = p0
        u_opt = np.zeros((2, ))

        for i in range(0, N):
            
            phi_ego = turn_specs(B1, N-i, 'ego')
            if mode == 0:
                phi = phi_ego
            elif mode == 2:
                oppo_std, pedes_std = scenario(sys, tr, samples, Q, i)
                phi_oppo = safety_specs_multi_modal(B1, N-i, std=oppo_std, dist=4, sys_id='oppo')
                phi_pedes = safety_specs_multi_modal(B2, N-i, std=pedes_std, dist=2, sys_id='pedes')
                phi = phi_ego & phi_oppo & phi_pedes
            else:
                phi_oppo = safety_specs(B1, N-i, dist=4, sys_id='oppo')
                phi_pedes = safety_specs(B2, N-i, dist=2, sys_id='pedes')
                phi = phi_ego & phi_oppo & phi_pedes

            sys['ego'].x0 = tr['ego'][:, i, j]
            sys['oppo'].x0 = tr['oppo'][:, i, j]
            sys['pedes'].x0 = tr['pedes'][:, i, j]

            sys['ego'].param = np.array([0, l, 1])
            sys['oppo'].param = np.array(samples['oppo_simulate'][:, i, j])
            sys['pedes'].param = np.array(samples['pedes_simulate'][:, i, j])

            sys['ego'].update_matrices()
            sys['oppo'].update_matrices()
            sys['pedes'].update_matrices()

            solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
            solver.AddQuadraticCost(R)
            x, u, rho, _ = solver.Solve()
            
            if rho >= 0:
                u_opt = u[:, 0]
                
            tr['ego'][:, i + 1, j] = sys['ego'].f(tr['ego'][:, i, j], u_opt)
            tr['oppo'][:, i + 1, j] = sys['oppo'].f(tr['oppo'][:, i, j], v1[:, i])
            tr['pedes'][:, i + 1, j] = sys['pedes'].f(tr['pedes'][:, i, j], v2[:, i])
            
        np.save('results/case_2/xx_' + 'ego' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', tr['ego'][:, :, j])
        np.save('results/case_2/xx_' + 'oppo' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', tr['oppo'][:, :, j])
        np.save('results/case_2/xx_' + 'pedes' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy', tr['pedes'][:, :, j])
        
else:

    for j in range(0, M):
        tr['ego'][:, :, j] = np.load('results/case_2/xx_' + 'ego' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')
        tr['oppo'][:, :, j] = np.load('results/case_2/xx_' + 'oppo' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')
        tr['pedes'][:, :, j] = np.load('results/case_2/xx_' + 'pedes' + '_mode_' + str(mode) + '_seed_' + str(j) + '_c.npy')

sys['oppo'].x0 = o0
sys['pedes'].x0 = p0
cursors = [22, 26]
visualize(sys, tr, cursor=cursors[0], mode=mode)
