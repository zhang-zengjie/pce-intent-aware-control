import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from config.intersection.params import sys, phi, N, mode


u_opt = np.zeros((2, ))
solver = PCEMICPSolver(phi, sys, N)

for i in range(N):

    solver.syses['ego'].update_matrices(i)
    solver.syses['oppo'].update_matrices(i)
    solver.syses['pedes'].update_matrices(i)

    solver.syses['oppo'].predict_pce(i, N)
    solver.syses['oppo'].predict(i, N)
    solver.syses['pedes'].predict_pce(i, N)
    solver.syses['pedes'].predict(i, N)

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
        u_opt[1] = - solver.syses['ego'].states[3, i]/solver.syses['ego'].dt
    
    solver.syses['ego'].apply_control(i, u_opt)
    solver.syses['oppo'].apply_control(i, solver.syses['oppo'].useq[:, i])
    solver.syses['pedes'].apply_control(i, solver.syses['pedes'].useq[:, i])

np.save('results/case_2/x_mode_' + str(mode) + '.npy', solver.syses['ego'].states)
np.save('results/case_2/z_oppo_mode_' + str(mode) + '.npy', solver.syses['oppo'].pce_coefs)
np.save('results/case_2/z_pedes_mode_' + str(mode) + '.npy', solver.syses['pedes'].pce_coefs)
