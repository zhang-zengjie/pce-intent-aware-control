import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_2_config import *


Ts = 0.5            # The baseline value of sampling time delta_t
l = 4               # The baseline value of the vehicle length
N = 36              # The control horizon
M = 100               # Runs
R = np.array([[1, 0], [0, 50]])
Q = 25              # Scenario numbers

mode = 1    # Select simulation mode: 
            # 0 for no_reaction 
            # 1 for reaction with proposed method
            # 2 for reaction with conventional method

np.random.seed(7)

v1, v2 = get_intentions(N)          # The assumed control mode of the opponent vehicle (OV)
B1, B2 = gen_bases()                # Generate PCE bases
e0, o0, p0 = get_initials()         # Get initial conditions

sys = {'ego': BicycleModel(Ts, name='ego'),                                     # Dynamic model of the ego vehicle (EV)
       'oppo': BicycleModel(Ts, useq=v1, basis=B1, pce=True, name='oppo'),      # Dynamic model of the opponent vehicle (OV)
       'pedes': BicycleModel(Ts, useq=v2, basis=B2, pce=True, name='pedes')}    # Dynamic model of the pedestrian (PD)

spec_samples = {'oppo': sys['oppo'].basis.eta.sample([N, Q]),               # Samples for scenario of the opponent vehicle (OV)
                'pedes': sys['pedes'].basis.eta.sample([N, Q])}             # Samples for scenario of the pedestrian (PD)

tr = {'ego': np.zeros([sys['ego'].n, N + 1]),                                # Simulated trajectories of the ego vehicle (EV)
      'oppo': np.zeros([sys['oppo'].basis.L, sys['oppo'].n, N + 1]),         # Simulated trajectories of the opponent vehicle (OV)
      'pedes': np.zeros([sys['pedes'].basis.L, sys['pedes'].n, N + 1])}      # Simulated trajectories of the pedestrian (PD)

if True:

    tr['ego'][:, 0] = e0
    tr['oppo'][0, :, 0] = o0
    tr['pedes'][0, :, 0] = p0
    u_opt = np.zeros((2, ))

    sys['ego'].update_param(np.array([0, l, 1]))
    sys['oppo'].update_param(np.array([0, l, 1]))
    sys['pedes'].update_param(np.array([0, l, 1]))

    for i in range(0, N):
        
        # Update specification
        phi = get_spec(sys, tr, spec_samples, Q, N, i, mode)

        # Update current states and parameters
        sys['ego'].update_initial(tr['ego'][:, i])
        sys['oppo'].update_initial(tr['oppo'][0, :, i])
        sys['oppo'].update_initial_pce(tr['oppo'][:, :, i])
        sys['pedes'].update_initial(tr['pedes'][0, :, i])
        sys['pedes'].update_initial_pce(tr['pedes'][:, :, i])

        sys['ego'].update_matrices()
        sys['oppo'].update_matrices()
        sys['pedes'].update_matrices()

        # Solve
        solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
        solver.AddQuadraticCost(R)
        x, u, rho, _ = solver.Solve()
        
        # In case infeasibility
        if rho >= 0:
            u_opt = u[:, 0]
        else:
            u_opt[0] = 0
            u_opt[1] = -tr['ego'][3, i]/Ts
            
        # Probabilistic prediction
            
        tr['oppo'][:, :, i + 1] = sys['oppo'].predict_pce(1)[:, :, 1]
        tr['pedes'][:, :, i + 1] = sys['pedes'].predict_pce(1)[:, :, 1]

        # Simulate the next step

        tr['ego'][:, i + 1] = sys['ego'].f(tr['ego'][:, i], u_opt)
        tr['oppo'][0, :, i + 1] = sys['oppo'].f(tr['oppo'][0, :, i], sys['oppo'].useq[:, i])
        tr['pedes'][0, :, i + 1] = sys['pedes'].f(tr['pedes'][0, :, i], sys['pedes'].useq[:, i])
        
    np.save('results/case_2/xx_' + 'ego' + '_mode_' + str(mode) + '_seed_c.npy', tr['ego'])
    np.save('results/case_2/xx_' + 'oppo' + '_mode_' + str(mode) + '_seed_c.npy', tr['oppo'])
    np.save('results/case_2/xx_' + 'pedes' + '_mode_' + str(mode) + '_seed_c.npy', tr['pedes'])
        
else:

    tr['ego'] = np.load('results/case_2/xx_' + 'ego' + '_mode_' + str(mode) + '_seed_c.npy')

cursors = [24, 26]

tr_oppo_s = np.zeros([M, sys['oppo'].n, N + 1])
tr_pedes_s = np.zeros([M, sys['pedes'].n, N + 1])
samples_oppo = sys['oppo'].basis.eta.sample([M, ])
samples_pedes = sys['pedes'].basis.eta.sample([M, ])

for j in range(0, M):
    sys['oppo'].update_param(samples_oppo[:, j])
    sys['pedes'].update_param(samples_pedes[:, j])
    sys['oppo'].update_initial(o0)
    sys['pedes'].update_initial(p0)
    tr_oppo_s[j, :, :] = sys['oppo'].predict(N)
    tr_pedes_s[j, :, :] = sys['pedes'].predict(N)

visualize(tr['ego'], tr_oppo_s, tr_pedes_s, cursor=cursors[0])
