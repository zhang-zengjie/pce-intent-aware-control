import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from config.overtaking.params import ego, oppo, N, B, R, mode, v0, e0, o0
from config.overtaking.functions import gen_pce_specs
from libs.commons import model_checking


# Initialize the solver
sys = {ego.name: ego,
       oppo.name: oppo}

u_opt = np.zeros((2, ))

xx = np.zeros([ego.n, N + 1])
zz = np.zeros([oppo.basis.L, oppo.n, N + 1])

v0 = 10
xx[:, 0] = e0      
zz[0, :, 0] = o0          

phi = gen_pce_specs(B, N, v0*1.2, 'oppo')

solver = PCEMICPSolver(phi, sys, N, robustness_cost=True)

for i in range(N):
    
    # Update specification
    
    solver.syses["ego"].update_matrices(i)
    solver.syses["oppo"].update_matrices(i)

    solver.syses["oppo"].predict(i, N)
    solver.syses["oppo"].predict_pce(i, N)

    # Solve
    solver.RemoveDynamicsConstraints()
    solver.AddDynamicsConstraints(i)
    
    x, u, rho, _ = solver.Solve()

    # In case infeasibility
    if rho >= 0:
        u_opt = u[:, 0]

    solver.syses["ego"].update_measurements(i, u_opt)
    solver.syses["oppo"].update_measurements(i, oppo.useq[:, i])


np.save('results/case_1/x_mode_' + str(mode) + '.npy', solver.syses["ego"].states)
np.save('results/case_1/z_mode_' + str(mode) + '.npy', solver.syses["oppo"].pce_coefs)
