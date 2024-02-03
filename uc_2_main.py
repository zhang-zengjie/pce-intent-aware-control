import numpy as np
from libs.micp_pce_solvern import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from stlpy.systems.linear import DoubleIntegrator
from config.uc_2_config import visualize3D, visualize, turn_specs, safety_specs
from config.uc_2_config import l as lane
import math
from libs.pce_basis import PCEBasis

import chaospy as cp

# The assumed control input of the obstacle vehicle (OV)

Ts = 0.5    # The baseline value of sampling time delta_t
l = 4             # The baseline value of the vehicle lengthS
q = 2                       # The polynomial order
N = 30                      # The control horizon
M = 1
R = np.array([[10, 0], [0, 500]])

np.random.seed(7)

# The assumed control mode of the obstacle vehicle (OV)

gamma1 = np.linspace(0, 0, N)
a1 = np.linspace(0, -1.2, N)
u1 = np.array([gamma1, a1])

gamma2 = np.linspace(0, 0, N)
a2 = np.linspace(0, 0.5, N)
u2 = np.array([gamma2, a2])

# Generate the PCE instance and the specification
bias1 = cp.Normal(0, 1e-2)
length1 = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
intent1 = cp.DiscreteUniform(-1, 1)
eta1 = cp.J(bias1, length1, intent1) # Generate the random variable instance
B1 = PCEBasis(eta1, q)

bias2 = cp.Normal(0, 0.01)
length2 = cp.Uniform(lower=0.5-1e-3, upper=0.5+1e-3)
# intent_p = cp.Binomial(1, 0.3)
intent2 = cp.DiscreteUniform(0, 1)
eta2 = cp.J(bias2, length2, intent2) # Generate the random variable instance
B2 = PCEBasis(eta2, q)

e0 = np.array([-lane*2, -lane/2, 0, 0.5])            # Initial position of the ego vehicle (EV)
o0 = np.array([95, lane/2, math.pi, 8])           # Initial position of the obstacle vehicle (OV)
p0 = np.array([1.2*lane, 1.2*lane, math.pi, 0])

ego = BicycleModel(e0, [0, l, 1], Ts, name="ego", color='red')                  # Dynamic model of the ego vehicle (EV)
oppo = BicycleModel(o0, [0, l, 1], Ts, useq=u1, basis=B1, pce=True, name="oppo", color=(0, 0, 0.5))     # Dynamic model of the obstacle vehicle (OV)
pedes = BicycleModel(p0, [0, l, 1], Ts, useq=u2, basis=B2, pce=True, name="pedes", color=(1, 0.6, 0.2))

sys = {ego.name: ego,
       oppo.name: oppo,
       pedes.name: pedes}

if False:
    
    j = 0
    xx_ego = np.zeros([ego.n, N + 1, M])
    xx_oppo = np.zeros([oppo.n, N + 1, M])
    xx_pedes = np.zeros([pedes.n, N + 1, M])
    nodes_o = oppo.basis.eta.sample([N, M])
    nodes_p = pedes.basis.eta.sample([N, M])
        
    xx_ego[:, 0, j] = e0
    xx_oppo[:, 0, j] = o0
    xx_pedes[:, 0, j] = p0

    u_opt = np.zeros((2, ))

    for i in range(0, N):
        
        ego.x0 = xx_ego[:, i, j]
        oppo.x0 = xx_oppo[:, i, j]
        pedes.x0 = xx_pedes[:, i, j]

        oppo.param = np.array(nodes_o[:, i, j])
        pedes.param = np.array(nodes_p[:, i, j])

        ego.update_matrices()
        oppo.update_matrices()
        pedes.update_matrices()

        phi_oppo = safety_specs(B1, N-i, "oppo")
        phi_pedes = safety_specs(B2, N-i, "pedes")
        phi_ego = turn_specs(B1, N-i, "ego")
        phi = phi_ego & phi_oppo & phi_pedes
        solver = PCEMICPSolver(phi, sys, N-i, robustness_cost=True)
        
        solver.AddQuadraticCost(R)
        x, u, rho, _ = solver.Solve()
        
        if rho >= 0:
            u_opt = u[:, 0]
        xx_ego[:, i + 1, j] = ego.f(xx_ego[:, i, j], u_opt)
        xx_oppo[:, i + 1, j] = oppo.f(xx_oppo[:, i, j], u1[:, i])
        xx_pedes[:, i + 1, j] = pedes.f(xx_pedes[:, i, j], u2[:, i])
        
    
    np.save('results/case_2/xx_ego_seed_' + str(j) + '_c.npy', xx_ego[:, :, j])
    np.save('results/case_2/xx_oppo_seed_' + str(j) + '_c.npy', xx_oppo[:, :, j])
    np.save('results/case_2/xx_pedes_seed_' + str(j) + '_c.npy', xx_pedes[:, :, j])
    
else:

    xx_ego = np.zeros([ego.n, N + 1, M])
    xx_oppo = np.zeros([oppo.n, N + 1, M])
    xx_pedes = np.zeros([pedes.n, N + 1, M])

    j = 0
    xx_ego[:, :, j] = np.load('results/case_2/xx_ego_seed_' + str(j) + '_c.npy')
    xx_oppo[:, :, j] = np.load('results/case_2/xx_oppo_seed_' + str(j) + '_c.npy')
    xx_pedes[:, :, j] = np.load('results/case_2/xx_pedes_seed_' + str(j) + '_c.npy')


# oz = solver.predict["oppo"]
# pz = solver.predict["pedes"]
# model_checking(x, z, phi, 0)

# np.save('x.npy', x)

# x = np.load('x.npy')
# visualize3D(x, [oppo, pedes])
visualize(xx_ego[:, :, 0], [oppo, pedes], cursor=24)
