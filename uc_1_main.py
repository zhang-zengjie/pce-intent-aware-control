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

for i in range(0, N):
    
    # Update specification
    phi = gen_pce_specs(B, N-i, v0*1.2, 'oppo')

    # Update current states and parameters
    ego.update_initial(xx[:, i])
    oppo.update_initial(zz[0, :, i])
    oppo.update_initial_pce(zz[:, :, i])

    ego.update_matrices()
    oppo.update_matrices()

    # Solve
    solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
    solver.AddQuadraticCost(R)
    x, u, rho, _ = solver.Solve()

    # In case infeasibility
    if rho >= 0:
        u_opt = u[:, 0]

    # Probabilistic prediction
    zz[:, :, i + 1] = oppo.predict_pce(1)[:, :, 1]

    # Simulate the next step
    xx[:, i + 1] = ego.f(xx[:, i], u_opt)
    zz[0, :, i + 1] = oppo.f(zz[0, :, i], oppo.useq[:, i])

np.save('results/case_1/x_mode_' + str(mode) + '.npy', xx)
np.save('results/case_1/z_mode_' + str(mode) + '.npy', zz)
