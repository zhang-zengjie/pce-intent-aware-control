import numpy as np
import matplotlib.pyplot as plt


lanes = {'right': 1,
         'slow': 3,
         'middle': 5,
         'fast': 7,
         'left': 9}

x_range_dict = {'switch_lane': [-5, 165],
           'constant_speed': [25, 185],
           'speed_up': [-5, 165],
           'big_variance': [-5, 165]}

legend_loc_dict = {'switch_lane': (0.74, 0.15),
              'constant_speed': (0.74, 0.65),
              'speed_up': (0.74, 0.65),
              'big_variance': (0.74, 0.65)}

eps = 0.05          # Probability threshold
d_safe_x = 10
d_safe_y = 2

# Coefficients of the predicates
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])

def gen_pce_specs(B, N, v_lim, var_lim, sys_id):

    mu_safe = B.probability_formula(a2, o, lanes['right'] + d_safe_y/2, eps, sys_id) | \
        B.probability_formula(-a2, o, - lanes['left'] - d_safe_y/2, eps, sys_id) | \
        B.probability_formula(a1, -a1, d_safe_x, eps, sys_id) | \
        B.probability_formula(-a1, a1, d_safe_x, eps, sys_id) | \
        B.probability_formula(a2, -a2, d_safe_y, eps, sys_id) | \
        B.probability_formula(-a2, a2, d_safe_y, eps, sys_id)

    mu_belief = B.variance_formula(a1, var_lim, sys_id) & \
        B.expectation_formula(o, -a2, -lanes['middle'], sys_id) & \
        B.expectation_formula(o, -a4, -v_lim, sys_id)
    neg_mu_belief = B.neg_variance_formula(a1, var_lim, sys_id) | \
        B.expectation_formula(o, a2, lanes['middle'], sys_id) | \
        B.expectation_formula(o, a4, v_lim, sys_id)

    mu_overtake = B.probability_formula(a2, o, lanes['slow'] - 0.1, eps, sys_id).always(0, 3) & \
        B.probability_formula(-a2, o, - lanes['slow'] - 0.1, eps, sys_id).always(0, 3) & \
        B.probability_formula(a1, -a1, d_safe_y, eps, sys_id).always(0, 3) & \
        B.probability_formula(a3, o, - 1e-1, eps, sys_id).always(0, 3) & \
        B.probability_formula(-a3, o, - 1e-1, eps, sys_id).always(0, 3) 

    phi_safe = mu_safe.always(0, N)
    phi_belief = mu_belief.always(0, N)
    phi_neg_belief = neg_mu_belief.eventually(0, N)
    phi_overtake = mu_overtake.eventually(0, N-3)

    phi = (phi_neg_belief | phi_overtake) & phi_safe

    return phi


def visualize(x, z, T, mode):

    x_range=x_range_dict[mode]
    y_range=[0, 10]
    legend_loc=legend_loc_dict[mode]

    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(5.4, 1.4))
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.32)

    plt.plot(np.arange(x_range[0], x_range[1]), lanes['left'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['middle'] * np.ones((x_range[1] - x_range[0], )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['right'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')

    # Plot the trajectory of the ego vehicle (EV)
    for j in range(0, T):
        pev = ax.add_patch(Rectangle(xy=(x[0, j]-4, x[1, j]-0.5), width=4, height=1, angle=x[2, j]/3.14*180, linewidth=1, 
                               edgecolor='red', fill=True, facecolor=(255/255, 1-j/(T*2-2), 1-j/(T*2-2)), zorder=10))
        pov = ax.add_patch(Rectangle(xy=(z[0, j]-4, z[1, j]-0.5), width=4, height=1, angle=z[2, j]/3.14*180, linewidth=1, 
                               edgecolor='blue', fill=True, facecolor=(1-j/(T*2-2), 1-j/(T*4-4), 255/255), zorder=10))

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.legend([pev, pov], ['EV', 'OV'], loc=legend_loc, fontsize="8", ncol=2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.show()
