import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_2_config import *


Ts = 0.5            # The baseline value of sampling time delta_t
l = 4               # The baseline value of the vehicle length
N = 30              # The control horizon
M = 1               # Runs
R = np.array([[10, 0], [0, 500]])
Q = 25              # Scenario numbers

mode = 2    # Select simulation mode: 
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

tr = {'ego': np.zeros([sys['ego'].n, N + 1, M]),                                # Simulated trajectories of the ego vehicle (EV)
      'oppo': np.zeros([sys['oppo'].n, N + 1, M]),                              # Simulated trajectories of the opponent vehicle (OV)
      'pedes': np.zeros([sys['pedes'].n, N + 1, M])}                            # Simulated trajectories of the pedestrian (PD)

samples = {'oppo_simulate': sys['oppo'].basis.eta.sample([N, M]),               # Samples for simulate of the opponent vehicle (OV)
           'pedes_simulate': sys['pedes'].basis.eta.sample([N, M]),             # Samples for simulate of the pedestrian (PD)
           'oppo_predict': sys['oppo'].basis.eta.sample([N, M]),                # Samples for prediction of the opponent vehicle (OV)
           'pedes_predict': sys['pedes'].basis.eta.sample([N, M]),              # Samples for prediction of the pedestrian (PD)
           'oppo_scenario': sys['oppo'].basis.eta.sample([N, Q]),               # Samples for scenario of the opponent vehicle (OV)
           'pedes_scenario': sys['pedes'].basis.eta.sample([N, Q])}             # Samples for scenario of the pedestrian (PD)

if True:

    for j in range(0, M):

        tr['ego'][:, 0, j] = e0
        tr['oppo'][:, 0, j] = o0
        tr['pedes'][:, 0, j] = p0
        u_opt = np.zeros((2, ))

        for i in range(0, N):
            
            # Update specification
            phi = get_spec(sys, tr, samples, Q, N, i, j, mode)

            # Update current states and parameters
            sys['ego'].x0 = tr['ego'][:, i, j]
            sys['oppo'].x0 = tr['oppo'][:, i, j]
            sys['pedes'].x0 = tr['pedes'][:, i, j]

            sys['ego'].param = np.array([0, l, 1])
            sys['oppo'].param = np.array(samples['oppo_predict'][:, i, j])
            sys['pedes'].param = np.array(samples['pedes_predict'][:, i, j])

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
                
            # Simulate the next step
            sys['ego'].param = np.array([0, l, 1])
            sys['oppo'].param = np.array(samples['oppo_simulate'][:, i, j])
            sys['pedes'].param = np.array(samples['pedes_simulate'][:, i, j])

            tr['ego'][:, i + 1, j] = sys['ego'].f(tr['ego'][:, i, j], u_opt)
            tr['oppo'][:, i + 1, j] = sys['oppo'].f(tr['oppo'][:, i, j], sys['oppo'].useq[:, i])
            tr['pedes'][:, i + 1, j] = sys['pedes'].f(tr['pedes'][:, i, j], sys['pedes'].useq[:, i])
            
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
