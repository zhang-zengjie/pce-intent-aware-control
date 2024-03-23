import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from config.overtaking.params import initialize

# First of first, choose the mode
mode = 2    # Select intention mode: 
            # 0 for switching-lane OV 
            # 1 for constant-speed OV
            # 2 for speeding-up OV
N = 15      # Control horizon

# Initialize system and specification
sys, phi = initialize(mode, N)
            # sys: the dictionary of agents
            # phi: the task specification

# Load the solver
solver = PCEMICPSolver(phi, sys, N)
u_opt = np.zeros((2, ))

for i in range(N):
    
    # Update the linearized matrices
    solver.syses['ego'].update_matrices(i)
    solver.syses['oppo'].update_matrices(i)

    # Update the linearized prediction
    solver.syses['oppo'].predict(i, N)
    
    # Update the dynamics constraints
    solver.AddDynamicsConstraints(i)

    # Update the cost
    solver.cost = 0.0
    solver.AddRobustnessConstraint()
    solver.AddQuadraticCost(i)

    # Solve the problem
    x, u, rho, _ = solver.Solve()
                # x: the state decision variables
                # u: the control decision variables
                # rho: the specification satisfaction variable

    # Remove old dynamics constraints
    solver.RemoveDynamicsConstraints()

    # In case infeasibility, use the previous control input
    if (rho is not None) & (rho >=0):
        u_opt = u[:, i]

    # Apply the control input
    solver.syses['ego'].apply_control(i, u_opt)
    solver.syses['oppo'].apply_control(i, solver.syses['oppo'].useq[:, i])

# Save data
np.save('data/overtaking/x_mode_' + str(mode) + '.npy', solver.syses['ego'].states)
np.save('data/overtaking/z_mode_' + str(mode) + '.npy', solver.syses['oppo'].pce_coefs)
