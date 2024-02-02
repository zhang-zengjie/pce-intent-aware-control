import numpy as np
import matplotlib.pyplot as plt


lanes = {'right': 1,
         'slow': 3,
         'middle': 5,
         'fast': 7,
         'left': 9}

eps = 0.05          # Probability threshold
# v_lim = 30          # Velocity limit

# Coefficients of the predicates
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])

b = 5

def gen_pce_specs(B, N, v_lim, sys_id):

    mu_safe = B.probability_formula(a1, -a1, 10, eps, sys_id) | \
        B.probability_formula(-a1, a1, 10, eps, sys_id) | \
        B.probability_formula(a2, -a2, 2, eps, sys_id) | \
        B.probability_formula(-a2, a2, 2, eps, sys_id)

    mu_belief = B.variance_formula(a1, 12, sys_id) & \
        B.expectation_formula(o, -a2, -lanes['middle'], sys_id) & \
        B.expectation_formula(o, -a4, -v_lim, sys_id)
    neg_mu_belief = B.neg_variance_formula(a1, 12, sys_id) | \
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


def visualize(x, z, x_range, y_range, t_end):

    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(5.4, 1.4))

    plt.plot(np.arange(x_range[0], x_range[1]), lanes['left'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['middle'] * np.ones((x_range[1] - x_range[0], )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['right'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')

    # Plot the trajectory of the ego vehicle (EV)
    for j in range(0, t_end):
        pev = ax.add_patch(Rectangle(xy=(x[0, j]-4, x[1, j]-0.5), width=4, height=1, angle=x[2, j]/3.14*180, linewidth=1, 
                               edgecolor='red', fill=True, facecolor=(255/255, 1-j/(t_end*2-2), 1-j/(t_end*2-2)), zorder=10))
        pov = ax.add_patch(Rectangle(xy=(z[0, j]-4, z[1, j]-0.5), width=4, height=1, angle=z[2, j]/3.14*180, linewidth=1, 
                               edgecolor='blue', fill=True, facecolor=(1-j/(t_end*2-2), 1-j/(t_end*4-4), 255/255), zorder=10))

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('(m)')
    plt.ylabel('(m)')
    plt.legend([pev, pov], ['EV', 'OV'], loc=(0.7, 0.15), fontsize="8", ncol=2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.show()
