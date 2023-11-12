import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from libs.pce_basis import PCEBasis


lanes = {'right': 0,
         'slow': 2,
         'middle': 4,
         'fast': 6,
         'left': 8}


def gen_pce_specs(base_sampling_time, base_length, q, N):

    np.random.seed(7)

    # Sample delta_t
    delta_t = cp.Trunc(cp.Normal(base_sampling_time, 0.05), lower=base_sampling_time - 0.05, upper=base_sampling_time + 0.05)

    # Sample length
    length = cp.Trunc(cp.Normal(base_length, 0.1), lower=base_length - 0.1, upper=base_length + 0.1)
    
    eta = cp.J(delta_t, length) # Generate the random variable instance

    B = PCEBasis(eta, q)        # Initialize the PCE instance

    eps = 0.05          # Probability threshold
    v_lim = 30          # Velocity limit

    # Coefficients of the predicates
    o = np.zeros((4, ))

    a1 = np.array([1, 0, 0, 0])
    c1 = np.array([-1, 0, 0, 0])

    a2 = np.array([-1, 0, 0, 0])
    c2 = np.array([1, 0, 0, 0])

    a3 = np.array([0, 1, 0, 0])
    c3 = np.array([0, -1, 0, 0])

    a4 = np.array([0, 0, 0, 1])
    c4 = np.array([0, 0, 0, -1])

    a5 = np.array([0, 0, 1, 0])
    c5 = np.array([0, 0, -1, 0])

    b = 5

    mu_safe = B.probability_formula(a1, c1, b, eps) | B.probability_formula(a2, c2, b, eps) | B.probability_formula(a3, c3, b, eps)

    neg_mu_belief = B.neg_variance_formula(a3, 0.9) | B.expectation_formula(o, a3, lanes['middle']) | B.expectation_formula(o, a4, v_lim)

    mu_overtake = B.expectation_formula(a3, o, lanes['slow'] - 0.01) & B.expectation_formula(c3, o, - lanes['slow'] - 0.011) \
                    & B.expectation_formula(a1, c1, 2*b) \
                    & B.expectation_formula(a5, o, - 0.000001).always(0, 3) & B.expectation_formula(c5, o, - 0.000001).always(0, 3)

    phi_safe = mu_safe.always(0, N)
    phi_belief = neg_mu_belief.eventually(0, N)
    phi_overtake = mu_overtake.eventually(0, N-3)

    phi = phi_overtake

    phi = phi_belief | phi_overtake

    return B, phi


def visualize(x, z0, v, B, bicycle):

    from matplotlib.patches import Rectangle

    N = x.shape[1]-1
    H = 500

    fig, ax = plt.subplots()

    ax.plot(lanes['left'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')
    ax.plot(lanes['middle'] * np.ones((H, )), linestyle='dashed', linewidth=1, color='black')
    ax.plot(lanes['right'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')

    # Plot the trajectory of the ego vehicle (EV)
    p = ax.plot(x[0, :], x[1, :], linestyle='solid', linewidth=2, color='red')

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

    # Plot the trajectories of the obstacle vehicle (OV) 
    for i in range(M):
        ax.plot(mc_samples_linear[i, :, 0], mc_samples_linear[i, :, 1])
        ax.add_patch(Rectangle(xy=(mc_samples_linear[i, -1, 0]-4, mc_samples_linear[i, -1, 1]-1) ,width=4, height=2, linewidth=1, color='blue', fill=False))


    plt.xlim([0, H])
    # plt.ylim([0, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend([p1, p2], ['ego', 'obstacle'], loc='lower left')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # plt.figure()
    # pu, = plt.plot(np.arange(0, N+1), u[0])

    plt.show()
