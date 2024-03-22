import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from config.overtaking.params import ego, oppo, N, B, mode, v0
from config.overtaking.functions import gen_pce_specs
from libs.commons import model_checking
from gurobipy import GRB


# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

u_opt = np.zeros((2, ))

phi = gen_pce_specs(B, N, v0*1.2, 'oppo')

solver = PCEMICPSolver(phi, sys, N, robustness_cost=True)

for i in range(N):
    
    # Update specification
    
    solver.syses["ego"].update_matrices(i)
    solver.syses["oppo"].update_matrices(i)
    solver.syses['oppo'].predict_pce(i, N)
    solver.syses['oppo'].predict(i, N)

    # Solve
    
    solver.AddDynamicsConstraints(i)

    solver.cost = 0.0
    
    # solver.AddRobustnessCost()
    solver.AddRobustnessConstraint()
    solver.AddQuadraticCost(i)
    x, u, rho, _ = solver.Solve()

    solver.RemoveDynamicsConstraints()
    # In case infeasibility
    if (rho is not None) & (rho >=0):
        u_opt = u[:, i]

    solver.syses["ego"].apply_control(i, u_opt)
    solver.syses["oppo"].apply_control(i, oppo.useq[:, i])

    # solver.AddInputConstraint(i, u_opt)


np.save('results/case_1/x_mode_' + str(mode) + '.npy', solver.syses["ego"].states)
np.save('results/case_1/z_mode_' + str(mode) + '.npy', solver.syses["oppo"].pce_coefs)
