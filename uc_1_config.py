import numpy as np
import matplotlib.pyplot as plt
from libs.pce_basis import PCEBasis


lanes = {'right': 0,
         'slow': 2,
         'middle': 4,
         'fast': 6,
         'left': 8}


def gen_pce_specs(q, N, eta):

    B = PCEBasis(eta, q)        # Initialize the PCE instance

    eps = 0.05          # Probability threshold
    v_lim = 30          # Velocity limit

    # Coefficients of the predicates
    o = np.zeros((4, ))

    a1 = np.array([1, 0, 0, 0])
    a2 = np.array([0, 1, 0, 0])
    a3 = np.array([0, 0, 1, 0])
    a4 = np.array([0, 0, 0, 1])

    b = 5

    mu_safe = B.probability_formula(a1, -a1, 10, eps) | B.probability_formula(-a1, a1, 10, eps) | B.probability_formula(a2, -a2, 2, eps)

    mu_belief = B.variance_formula(a1, 20) & B.expectation_formula(o, -a2, -lanes['middle']) & B.expectation_formula(o, -a4, -v_lim)
    neg_mu_belief = B.neg_variance_formula(a1, 20) | B.expectation_formula(o, a2, lanes['middle']) | B.expectation_formula(o, a4, v_lim)

    mu_overtake = B.expectation_formula(a2, o, lanes['slow'] - 0.01) & B.expectation_formula(-a2, o, - lanes['slow'] - 0.011) \
                    & B.expectation_formula(a1, -a1, 2*b) \
                    & B.expectation_formula(a3, o, - 1e-6).always(0, 3) & B.expectation_formula(-a3, o, - 1e-6).always(0, 3) 

    phi_safe = mu_safe.always(0, N)
    phi_belief = mu_belief.always(0, N)
    phi_neg_belief = neg_mu_belief.eventually(0, N)
    phi_overtake = mu_overtake.eventually(0, N-3)

    phi = (phi_neg_belief | phi_overtake) & phi_safe

    return B, phi


def model_checking(x, z, spec, k):

    L = (1 + z.shape[0]) * z.shape[1]
    xx = np.zeros([L, z.shape[2]])

    for i in range(z.shape[2]):
        xx[:z.shape[1], i] = x[:, i]
        xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

    rho = spec.robustness(xx, k)

    return rho


def visualize(x, z0, v, B, bicycle):

    from matplotlib.patches import Rectangle

    N = x.shape[1]-1
    H = 600

    plt.figure(figsize=(5,2))

    plt.plot(lanes['left'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')
    plt.plot(lanes['middle'] * np.ones((H, )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(lanes['right'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')

    M = 64

    # Sample parameters from distribution eta
    nodes = B.eta.sample([M, ])

    # Generate the sampled trajectories of the obstacle vehicle (OV) 
    mc_samples_linear = np.zeros([M, N + 1, 4])
    mc_samples_linear[:, 0, :] = z0

    for i in range(M):
        bicycle.update_parameter(nodes[:, i])
        for j in range(N):
            mc_samples_linear[i, j + 1, :] = mc_samples_linear[i, j, :] + bicycle.Al @ mc_samples_linear[i, j, :] + bicycle.Bl @ v[:, j] + bicycle.El

    # Plot the trajectory of the ego vehicle (EV)
    tr1, = plt.plot(x[0, :], x[1, :], linestyle='solid', linewidth=2, color='red')
    p1, = plt.plot(x[0, -1], x[1, -1], alpha=0.8, color='red', marker="D", markersize=8)

    # Plot the trajectories of the obstacle vehicle (OV) 
    for i in range(M):
        tr2, = plt.plot(mc_samples_linear[i, :, 0], mc_samples_linear[i, :, 1], color=(0, 0, 0.5))
        # ax.add_patch(Rectangle(xy=(mc_samples_linear[i, -1, 0]-4, mc_samples_linear[i, -1, 1]-1) ,width=4, height=2, linewidth=1, color='blue', fill=False))
        p2, = plt.plot(mc_samples_linear[i, -1, 0]-4, mc_samples_linear[i, -1, 1], alpha=0.8, color=(0, 0, 0.5), marker="D", markersize=8)

    plt.xlim([0, H])
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend([tr1, p1, tr2, p2], ['ego trajectory', 'ego position', 'obstacle trajectory', 'obstacle position'], loc='upper right', fontsize="10", ncol=2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.show()
