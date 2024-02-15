import numpy as np
import matplotlib.pyplot as plt
import math
from libs.pce_basis import PCEBasis
import chaospy as cp


lanes = {'right': 1,
         'slow': 3,
         'middle': 5,
         'fast': 7,
         'left': 9}

mode_list = ['switch_lane', 'constant_speed', 'speed_up']

x_range_dict = {'switch_lane': [-5, 205],
                'constant_speed': [-5, 205],
                'speed_up': [-5, 205]}

legend_loc_dict = {'switch_lane': (0.74, 0.15),
                    'constant_speed': (0.74, 0.65),
                    'speed_up': (0.74, 0.65)}

eps = 0.05          # Probability threshold
d_safe_x = 10
d_safe_y = 2

# Coefficients of the predicates
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])


def get_intension(N, mode=1):

    if mode_list[mode] == 'switch_lane':  # That the OV is trying to switch to the fast lane
        gamma = np.array([0.005 * math.sin(i*6.28/(N-1)) for i in range(N)])
        a = np.zeros([N, ])

    elif mode_list[mode] == 'constant_speed':  # That the OV is trying to slow down (intention aware)
        gamma = np.linspace(0, 0, N)
        a = np.linspace(0, 0, N)

    elif mode_list[mode] == 'speed_up':   # That the OV is trying to speed_up (adversarial action)
        gamma = np.linspace(0, 0, N)
        a = np.linspace(0, 2, N)
        
    else:
        print('Mode number invalid. Abort...')
        exit()

    v = np.array([gamma, a])
    return v


def gen_bases(l):

    sigma = 0.1
    q = 2       # The polynomial order
    bias = cp.Trunc(cp.Normal(0, sigma), lower=-sigma, upper=sigma)
    length = cp.Uniform(lower=l - 1e-2, upper=l + 1e-2)
    intent = cp.Normal(1, 1e-3)
    eta = cp.J(bias, length, intent) # Generate the random variable instance
    B = PCEBasis(eta, q)

    return B

def gen_pce_specs(B, N, v_lim, sys_id):

    mu_safe_1 = B.gen_bs_predicate(a1, -a1, d_safe_x, eps, sys_id) | \
        B.gen_bs_predicate(-a1, a1, d_safe_x, eps, sys_id) | \
        B.gen_bs_predicate(a2, -a2, d_safe_y, eps, sys_id) | \
        B.gen_bs_predicate(-a2, a2, d_safe_y, eps, sys_id)

    mu_safe_2 = B.gen_bs_predicate(a2, o, lanes['right'] + d_safe_y/2, eps, sys_id) | \
        B.gen_bs_predicate(-a2, o, - lanes['left'] - d_safe_y/2, eps, sys_id)

    mu_belief = B.gen_bs_predicate(o, -a2, -lanes['middle'], 1, sys_id) & \
        B.gen_bs_predicate(o, -a4, -v_lim, 1, sys_id)
    neg_mu_belief = B.gen_bs_predicate(o, a2, lanes['middle'], 1, sys_id) | \
        B.gen_bs_predicate(o, a4, v_lim, 1, sys_id)

    mu_overtake = B.gen_bs_predicate(a2, o, lanes['slow'] - 0.1, eps, sys_id) & \
        B.gen_bs_predicate(-a2, o, - lanes['slow'] - 0.1, eps, sys_id) & \
        B.gen_bs_predicate(a1, -a1, d_safe_x, eps, sys_id) & \
        B.gen_bs_predicate(a3, o, - 1e-1, eps, sys_id) & \
        B.gen_bs_predicate(-a3, o, - 1e-1, eps, sys_id)

    mu_safe = mu_safe_1 & mu_safe_2
    phi_belief = mu_belief.always(0, N)
    phi_neg_belief = neg_mu_belief.eventually(0, N)

    if N > 3:
        phi = (phi_neg_belief | mu_overtake.always(0, 3).eventually(0, N-3)) & mu_safe.always(0, N)
    else:
        phi = (phi_neg_belief | mu_overtake.always(0, N)) & mu_safe.always(0, N)
    return phi


def visualize(x, z, mode):

    T = x.shape[1]
    M = z.shape[2]

    x_range=x_range_dict[mode_list[mode]]
    y_range=[0, 10]
    legend_loc=legend_loc_dict[mode_list[mode]]

    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(5.4, 1.4))
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.32)

    plt.plot(np.arange(x_range[0], x_range[1]), lanes['left'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['middle'] * np.ones((x_range[1] - x_range[0], )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['right'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')

    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')

    # Plot the trajectory of the ego vehicle (EV)
    for i in range(0, T - 1):
        for j in range(0, M):
            pov = ax.add_patch(Rectangle(xy=(z[0, i, j]-4, z[1, i, j]-0.5), width=4, height=1, angle=z[2, i, j]/3.14*180, linewidth=1, linestyle='dotted',
                                edgecolor=(0, 0, 51*4/255), fill=True, facecolor=c_oppo(i*0.8/T), zorder=10+i*M+z[1, i, j]))

        pev = ax.add_patch(Rectangle(xy=(x[0, i]-4, x[1, i]-0.5), width=4, height=1, angle=x[2, i]/3.14*180, linewidth=1, linestyle='dotted',
                                edgecolor=(51*4/255, 0, 0), fill=True, facecolor=c_ego(i*0.8/T), zorder=10+M*T+i*M+x[1, i]))
        
    i = T - 1

    for j in range(0, M):
        ax.add_patch(Rectangle(xy=(z[0, i, j]-4, z[1, i, j]-0.5), width=4, height=1, angle=z[2, i, j]/3.14*180, linewidth=1.5, 
                                edgecolor=(0, 0, 51/255), fill=True, facecolor=c_oppo(i*0.8/T), zorder=10+i*M+z[1, i, j]))

    ax.add_patch(Rectangle(xy=(x[0, i]-4, x[1, i]-0.5), width=4, height=1, angle=x[2, i]/3.14*180, linewidth=1.5, 
                                edgecolor=(51/255, 0, 0), fill=True, facecolor=c_ego(i*0.8/T), zorder=10+M*T+i*M+x[1, i]))

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.legend([pev, pov], ['EV', 'OV'], loc=legend_loc, fontsize="8", ncol=2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.show()
