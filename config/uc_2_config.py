import numpy as np
import matplotlib.pyplot as plt
from libs.pce_basis import PCEBasis
import math


l = 8 # The lane width

def oppo_specs(q, N, eta, sys_id):

    B = PCEBasis(eta, q)        # Initialize the PCE instance

    eps = 0.05          # Probability threshold
    v_lim = 30          # Velocity limit

    # Coefficients of the predicates
    o = np.zeros((4, ))

    a1 = np.array([1, 0, 0, 0])
    a2 = np.array([0, 1, 0, 0])
    a3 = np.array([0, 0, 1, 0])
    a4 = np.array([0, 0, 0, 1])
    
    mu_safe = B.probability_formula(a1, -a1, 4, eps, name=sys_id) | \
        B.probability_formula(-a1, a1, 4, eps, name=sys_id) | \
        B.probability_formula(a2, -a2, 4, eps, name=sys_id) | \
        B.probability_formula(-a2, a2, 4, eps, name=sys_id)
    
    reach = B.expectation_formula(a1, o, l/2 - 1e-2, name=sys_id).always(0, 3) & \
        B.expectation_formula(-a1, o, -l/2 - 1e-2, name=sys_id).always(0, 3) & \
        B.expectation_formula(a3, o, math.pi-1e-6, name=sys_id).always(0, 3) & \
        B.expectation_formula(-a3, o, -math.pi-1e-6, name=sys_id).always(0, 3)

    bet_out = B.expectation_formula(a2, o, -l/2 - 1e-2, name=sys_id) & \
        B.expectation_formula(-a2, o, l/2 - 1e-2, name=sys_id)
    vel_out = B.expectation_formula(a1, o, -1.2*l, name=sys_id)
    keep_out = vel_out | bet_out

    bet_in = B.expectation_formula(a1, o, l/2 - 1e-2, name=sys_id) & \
        B.expectation_formula(-a1, o, -l/2 - 1e-2, name=sys_id)
    vel_in = B.expectation_formula(-a2, o, -1.2*l, name=sys_id)
    keep_in = vel_in | bet_in

    phi = reach.eventually(0, N-3) & keep_out.always(0, N) & keep_in.always(0, N) & mu_safe.always(0, N)

    return B, phi


def pedes_specs(q, N, eta, sys_id):

    B = PCEBasis(eta, q)        # Initialize the PCE instance

    eps = 0.05          # Probability threshold
    v_lim = 30          # Velocity limit

    # Coefficients of the predicates
    o = np.zeros((4, ))

    a1 = np.array([1, 0, 0, 0])
    a2 = np.array([0, 1, 0, 0])
    a3 = np.array([0, 0, 1, 0])
    a4 = np.array([0, 0, 0, 1])
    
    mu_safe = B.probability_formula(a1, -a1, 2, eps, name=sys_id) | \
        B.probability_formula(-a1, a1, 2, eps, name=sys_id) | \
        B.probability_formula(a2, -a2, 2, eps, name=sys_id) | \
        B.probability_formula(-a2, a2, 2, eps, name=sys_id)

    phi = mu_safe.always(0, N)

    return B, phi


def model_checking(x, z, spec, k):

    L = (1 + z.shape[0]) * z.shape[1]
    xx = np.zeros([L, z.shape[2]])

    for i in range(z.shape[2]):
        xx[:z.shape[1], i] = x[:, i]
        xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

    rho = spec.robustness(xx, k)

    return rho



def visualize(x, oppos):

    plt.figure(figsize=(5,5))
    
    N = x.shape[1]-1

    gray = (102/255, 102/255, 102/255)
    light_gray = (230/255, 230/255, 230/255)
    # Draw the environment
    for i in [-1, 1]:

        for r in np.arange(-0.9, 1, 0.1):
            plt.plot([r*l, r*l], [1.1*i*l, 1.4*i*l], color=gray, linewidth=2)
            plt.plot([1.1*i*l, 1.4*i*l], [r*l, r*l], color=gray, linewidth=2)

        plt.plot([0, 0], [1.5*i*l, 3*i*l], color='black', linewidth=1)
        plt.plot([1.5*i*l, 3*i*l], [0, 0], color='black', linewidth=1)

        for j in [-1, 1]:
            plt.plot([i*l, i*l], [j*l, 3*j*l], color='black', linewidth=2)
            plt.plot([j*l, 3*j*l], [i*l, i*l], color='black', linewidth=2)

            plt.plot([0.5*i*l, 0.5*i*l], [1.5*j*l, 3*j*l], color=light_gray, linewidth=1, linestyle='dotted')
            plt.plot([1.5*j*l, 3*j*l], [0.5*i*l, 0.5*i*l], color=light_gray, linewidth=1, linestyle='dotted')

    # Plot the trajectory of the ego vehicle (EV)
    tr1, = plt.plot(x[0, :], x[1, :], linestyle='solid', linewidth=2, color='red')
    p1, = plt.plot(x[0, 22], x[1, 22], alpha=0.8, color='red', marker="D", markersize=8)

    M = 64

    for sys in oppos:
        # Sample parameters from distribution eta
        nodes_o = sys.basis.eta.sample([M, ])

        # Generate the sampled trajectories of the obstacle vehicle (OV) 
        
        mc_oppo = np.zeros([M, 4, N + 1])
        for i in range(M):
            # oppo.update_initial(z0)
            sys.update_parameter(nodes_o[:, i])
            mc_oppo[i] = sys.predict_linear(N)

        for i in range(M):
            tr2, = plt.plot(mc_oppo[i, 0, :], mc_oppo[i, 1, :], color=sys.color)
            # ax.add_patch(Rectangle(xy=(mc_oppo[i, -1, 0]-4, mc_oppo[i, -1, 1]-1) ,width=4, height=2, linewidth=1, color='blue', fill=False))
            p2, = plt.plot(mc_oppo[i, 0, 22], mc_oppo[i, 1, 22], alpha=0.8, color=sys.color, marker="D", markersize=8)
            p2, = plt.plot(mc_oppo[i, 0, 0], mc_oppo[i, 1, 0], alpha=0.8, color=sys.color, marker="*", markersize=18)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlim([-3*l, 3*l])
    plt.ylim([-3*l, 3*l])

    plt.show()
