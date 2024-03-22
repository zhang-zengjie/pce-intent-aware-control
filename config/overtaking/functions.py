import numpy as np
import matplotlib.pyplot as plt
import math
from libs.pce_basis import PCEBasis
import chaospy as cp
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import CubicSpline


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
d_safe_x = 5
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


    # phi = (phi_neg_belief | mu_overtake.always(0, 1).eventually(0, N-1)) & mu_safe.always(0, N)
    
    phi = mu_safe.always(0, N)

    return phi


def visualize(x, z, mode):

    T = x.shape[1]
    M = z.shape[2]

    x_range=x_range_dict[mode_list[mode]]
    y_range=[0, 10]
    legend_loc=legend_loc_dict[mode_list[mode]]

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


def record(x, z, mode, Ts=1, fps=12):

    T = x.shape[1]
    M = z.shape[2]

    ts = np.arange(0, T)
    tau = np.arange(0, T, 1/(fps*Ts))

    x_func_x = CubicSpline(ts, x[0])
    x_func_y = CubicSpline(ts, x[1])
    x_func_t = CubicSpline(ts, x[2])
    z_funcs_x = [CubicSpline(ts, z[0, :, j]) for j in range(0, M)]
    z_funcs_y = [CubicSpline(ts, z[1, :, j]) for j in range(0, M)]
    z_funcs_t = [CubicSpline(ts, z[2, :, j]) for j in range(0, M)]

    x_range=x_range_dict[mode_list[mode]]
    y_range=[0, 10]

    metadata = dict(title='Movie', artist='Zengjie Zhang')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(8, 4.5))

    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg-full_build\\bin\\ffmpeg.exe'

    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.32)

    plt.plot(np.arange(x_range[0], x_range[1]), lanes['left'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['middle'] * np.ones((x_range[1] - x_range[0], )), linestyle='dashed', linewidth=1, color='black')
    plt.plot(np.arange(x_range[0], x_range[1]), lanes['right'] * np.ones((x_range[1] - x_range[0], )), linestyle='solid', linewidth=2, color='black')

     #
    plt.ylim([0, 10])
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')

    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')

    with writer.saving(fig, 'highway_mode_' + str(mode) + '.mp4', 300):

        legend_loc=legend_loc_dict[mode_list[mode]]

        # Plot the trajectory of the ego vehicle (EV)
        for i, t in enumerate(tau):
            pov = [ax.add_patch(Rectangle(xy=(z_funcs_x[j](t)-4, z_funcs_y[j](t)-0.5), width=3.6, height=1.8, angle=z_funcs_t[j](t)/3.14*180, linewidth=1.5, linestyle='dotted',
                                    edgecolor=(0, 0, 0.2), fill=True, facecolor=c_oppo(i*0.8/len(tau)), zorder=10+i*M+z_funcs_y[j](t)))
                    for j in range(M)]

            pev = ax.add_patch(Rectangle(xy=(x_func_x(t)-4, x_func_y(t)-0.5), width=3.6, height=1.8, angle=x_func_t(t)/3.14*180, linewidth=1.5, linestyle='dotted',
                                    edgecolor=(0.2, 0, 0), fill=True, facecolor=c_ego(i*0.8/len(tau)), zorder=10+M*len(tau)+i*M+x_func_y(t)))
            
            if mode == 0:
                plt.xlim([0.8*i - 5,  0.8*i + 25])
            elif mode == 1:
                plt.xlim([1.1*i - 5,  1.1*i + 25])
            else:
                plt.xlim([0.8*i - 5,  0.8*i + 25])
            writer.grab_frame()
            print('Writing frame ' + str(i) + ' out of ' + str(len(tau)))
            pev.remove()
            for j in range(M):
                pov[j].remove() 

        # plt.legend([pev, pov], ['EV', 'OV'], loc=legend_loc, fontsize="8", ncol=2)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        