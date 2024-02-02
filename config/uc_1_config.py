import numpy as np
import matplotlib.pyplot as plt


lanes = {'right': 1,
         'slow': 3,
         'middle': 5,
         'fast': 7,
         'left': 9}

eps = 0.05          # Probability threshold
v_lim = 30          # Velocity limit

# Coefficients of the predicates
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])

b = 5

def gen_pce_specs(B, N, sys_id):

    mu_safe = B.probability_formula(a1, -a1, 10, eps, sys_id) | \
        B.probability_formula(-a1, a1, 10, eps, sys_id) | \
        B.probability_formula(a2, -a2, 2, eps, sys_id) | \
        B.probability_formula(-a2, a2, 2, eps, sys_id)

    mu_belief = B.variance_formula(a1, 20, sys_id) & \
        B.expectation_formula(o, -a2, -lanes['middle'], sys_id) & \
        B.expectation_formula(o, -a4, -v_lim, sys_id)
    neg_mu_belief = B.neg_variance_formula(a1, 20, sys_id) | \
        B.expectation_formula(o, a2, lanes['middle'], sys_id) | \
        B.expectation_formula(o, a4, v_lim, sys_id)

    mu_overtake = B.expectation_formula(a2, o, lanes['slow'] - 0.1, sys_id) & \
        B.expectation_formula(-a2, o, - lanes['slow'] - 0.1, sys_id) & \
        B.expectation_formula(a1, -a1, 2*b, sys_id) & \
        B.expectation_formula(a3, o, - 1e-1, sys_id).always(0, 3) & \
        B.expectation_formula(-a3, o, - 1e-1, sys_id).always(0, 3) 

    phi_safe = mu_safe.always(0, N)
    phi_belief = mu_belief.always(0, N)
    phi_neg_belief = neg_mu_belief.eventually(0, N)
    phi_overtake = mu_overtake.eventually(0, N-3)

    phi = (phi_neg_belief | phi_overtake) & phi_safe

    return phi


def visualize(x, byc, x_range, y_range, t_end):

    from matplotlib.patches import Rectangle

    N = t_end
    H = 600

    fig, ax = plt.subplots(figsize=(7,1))

    plt.plot(lanes['left'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')
    plt.plot(lanes['middle'] * np.ones((H, )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(lanes['right'] * np.ones((H, )), linestyle='solid', linewidth=2, color='black')

    M = 64

    # Sample parameters from distribution eta
    nodes = byc.basis.eta.sample([M, ])

    mc = np.zeros([M, 4, N + 1])
    for i in range(M):
        # byc.update_initial(z0)
        byc.param = np.array([nodes[0, i], nodes[1, i], 1])
        byc.update_lin_matrices()
        # 
        mc[i] = byc.predict_lin(N)

    # Plot the trajectory of the ego vehicle (EV)
    for j in range(0, t_end):
        # tr1, = plt.plot(x[0, :t_end], x[1, :t_end], linestyle='solid', linewidth=2, color='red')
        # p1, = plt.plot(x[0, j-1], x[1, j-1], alpha=0.8, color='red', marker="D", markersize=8)
        ax.add_patch(Rectangle(xy=(x[0, j]-4, x[1, j]-0.5), width=4, height=1, angle=x[2, j]/3.14*180, linewidth=1, edgecolor='red', fill=True, facecolor=(255/255, 1-j/(t_end*2-2), 1-j/(t_end*2-2)), zorder=10))

    # Plot the trajectories of the obstacle vehicle (OV) 
    for j in range(0, t_end):
        for i in range(M):
            # tr2, = plt.plot(mc[i, 0, :t_end], mc[i, 1, :t_end], color=(0, 0, 0.5))
            # ax.add_patch(Rectangle(xy=(mc[i, -1, 0]-4, mc[i, -1, 1]-1) ,width=4, height=2, linewidth=1, color='blue', fill=False))
            # p2, = plt.plot(mc[i, 0, j-1]-4, mc[i, 1, j-1], alpha=0.8, color=(0, 0, 0.5), marker="D", markersize=8)
            ax.add_patch(Rectangle(xy=(mc[i, 0, j]-4, mc[i, 1, j]-0.5), width=4, height=1, angle=x[2, j]/3.14*180, linewidth=1, edgecolor='blue', fill=True, facecolor=(1-j/(t_end*2-2), 1-j/(t_end*4-4), 255/255), zorder=j*(H-mc[i, 0, t_end-1])))

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend([tr1, p1, tr2, p2], ['ego trajectory', 'ego position', 'obstacle trajectory', 'obstacle position'], loc='upper right', fontsize="10", ncol=2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.show()
