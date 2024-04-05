import numpy as np
from config import initialize, data_dir
from draw import draw
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root)

from commons.pce_micp_solver import PCEMICPSolver

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def main(scene):

    N = 35
    
    print("---------------------------------------------------------")
    print('Initializing...')
    print("---------------------------------------------------------")
    # Initialize system and specification
    agents, phi = initialize(scene, N)
                # agents: the dictionary of agents
                    # agents['ego']: ego vehicle (EV)
                    # agents['oppo']: opponent vehicle (OV)
                    # agents['pedes']: pedestrians
                # phi: the task specification

    # Load the solver
    solver = PCEMICPSolver(phi, agents, N)
    runtime = np.zeros((N, ))
    u_opt = np.zeros((2, ))

    for i in range(N):

        # Update the linearized matrices
        solver.agents['ego'].update_matrices(i)
        solver.agents['oppo'].update_matrices(i)
        solver.agents['pedes'].update_matrices(i)

        # Update the linearized prediction
        solver.agents['oppo'].predict(i, N)
        solver.agents['pedes'].predict(i, N)

        # Update the dynamics constraints
        solver.AddDynamicsConstraints(i)

        # Update the cost
        solver.cost = 0.0
        solver.AddRobustnessCost()
        solver.AddQuadraticCost(i)

        # Solve the problem
        _, u, rho, runtime[i] = solver.Solve()
                    # x: the state decision variables
                    # u: the control decision variables
                    # rho: the specification satisfaction variable
        
        # Remove old dynamics constraints
        solver.RemoveDynamicsConstraints()

        # In case infeasibility, stop
        if (rho is not None) & (rho >=0):
            u_opt = u[:, i]
        else:
            u_opt[0] = 0
            u_opt[1] = - solver.agents['ego'].states[3, i]/solver.agents['ego'].dt
        
        # Apply the control input
        solver.agents['ego'].apply_control(i, u_opt)
        solver.agents['oppo'].apply_control(i, solver.agents['oppo'].useq[:, i])
        solver.agents['pedes'].apply_control(i, solver.agents['pedes'].useq[:, i])

    return solver.agents


if __name__ == "__main__":

    # First of first, choose the scenario
    scene = 0       # 0 for no awareness
                    # 1 for intention-aware
    # Choose the instant of the view
    step = 20       # 16: the step to show relation with the OV
                    # 20: the step to show relation with the pedestrian
    agents = main(scene)

    draw(agents, scene, step)