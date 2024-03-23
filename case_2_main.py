import numpy as np
from libs.pce_milp_solver import PCEMILPSolver
from config.intersection.params import initialize

# First of first, choose the scenario
scene = 0    # Select simulation scenario: 
        # 0 for no_awareness 
        # 1 for intention_aware
N = 35

# Initialize system and specification
agents, phi = initialize(scene, N)
            # agents: the dictionary of agents
                # agents['ego']: ego vehicle (EV)
                # agents['oppo']: opponent vehicle (OV)
                # agents['pedes']: pedestrians
            # phi: the task specification

u_opt = np.zeros((2, ))
solver = PCEMILPSolver(phi, agents, N)

for i in range(N):

    solver.agents['ego'].update_matrices(i)
    solver.agents['oppo'].update_matrices(i)
    solver.agents['pedes'].update_matrices(i)

    #solver.agents['oppo'].predict_pce(i, N)
    solver.agents['oppo'].predict(i, N)
    #solver.agents['pedes'].predict_pce(i, N)
    solver.agents['pedes'].predict(i, N)

    # Solve
    solver.AddDynamicsConstraints(i)

    solver.cost = 0.0
    solver.AddRobustnessCost()
    # solver.AddRobustnessConstraint()
    solver.AddQuadraticCost(i)
    x, u, rho, _ = solver.Solve()
    
    solver.RemoveDynamicsConstraints()

    # In case infeasibility, stop
    if (rho is not None) & (rho >=0):
        u_opt = u[:, i]
    else:
        u_opt[0] = 0
        u_opt[1] = - solver.agents['ego'].states[3, i]/solver.agents['ego'].dt
    
    solver.agents['ego'].apply_control(i, u_opt)
    solver.agents['oppo'].apply_control(i, solver.agents['oppo'].useq[:, i])
    solver.agents['pedes'].apply_control(i, solver.agents['pedes'].useq[:, i])

np.save('data/intersection/x_scene_' + str(scene) + '.npy', solver.agents['ego'].states)
np.save('data/intersection/z_oppo_scene_' + str(scene) + '.npy', solver.agents['oppo'].pce_coefs)
np.save('data/intersection/z_pedes_scene_' + str(scene) + '.npy', solver.agents['pedes'].pce_coefs)
