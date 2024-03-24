import numpy as np
import matplotlib.pyplot as plt
import math
from libs.pce_basis import PCEBasis
import chaospy as cp
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import CubicSpline
from libs.commons import tf_anchor as tf

# The latitudinal coordinates
lc = {'r': 1,         # right boundary
      's': 3,         # slow lane centerline
      'm': 5,         # middle line
      'f': 7,         # fast lane centerline
      'l': 9}         # left boundary

locs = ((0.74, 0.15), (0.74, 0.65), (0.74, 0.65))

eps = 0.05          # Probability threshold
dx = 4              # Longitudinal safe distance
dy = 2              # Latitudinal safe distance 
v0 = 10          
vl = v0*1.2         # Speed limit

# Coefficients of the predicates
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])


def get_feedforward(N, mode):

    if mode == 0:  # That the OV is trying to switch to the fast lane
        gamma = np.array([0.005 * math.sin(i*6.28/(N-1)) for i in range(N)])
        a = np.zeros((N, ))

    elif mode == 1:  # That the OV is trying to slow down (intention aware)
        gamma = np.zeros((N, ))
        a = - 0.25 * np.ones((N, ))

    elif mode == 2:   # That the OV is trying to speed_up (adversarial action)
        gamma = np.zeros((N, ))
        a = 0.5 * np.ones((N, ))
        
    else:
        print('Mode number invalid. Abort...')
        exit()

    v = np.array([gamma, a])
    return v


def get_initial_states():

    e0 = np.array([0, lc['f'], 0, v0*1.2]) # Initial position of the ego vehicle (EV)
    o0 = np.array([2*v0, lc['s'], 0, v0])  # Initial position of the obstacle vehicle (OV)

    return e0, o0


def get_bases(l, q):

    sigma = 0.1
    bias = cp.Trunc(cp.Normal(0, sigma), lower=-sigma, upper=sigma)
    length = cp.Uniform(lower=l - 1e-2, upper=l + 1e-2)
    intent = cp.Normal(1, 1e-3)         # Certain intention
    eta = cp.J(bias, length, intent)    # Generate the random variable instance
    B = PCEBasis(eta, q)

    return B


def get_specs(B, N, sys_id):

    mu_safe_1 = B.gen_bs_predicate(a1, -a1, dx, eps, sys_id) | \
        B.gen_bs_predicate(-a1, a1, dx, eps, sys_id) | \
        B.gen_bs_predicate(a2, -a2, dy, eps, sys_id) | \
        B.gen_bs_predicate(-a2, a2, dy, eps, sys_id)

    mu_safe_2 = B.gen_bs_predicate(a2, o, lc['r'] + dy/2, eps, sys_id) | \
        B.gen_bs_predicate(-a2, o, - lc['l'] - dy/2, eps, sys_id)

    mu_belief = B.gen_bs_predicate(o, -a2, -lc['m'], 1, sys_id) & \
        B.gen_bs_predicate(o, -a4, -vl, 1, sys_id)
    neg_mu_belief = B.gen_bs_predicate(o, a2, lc['m'], 1, sys_id) | \
        B.gen_bs_predicate(o, a4, vl, 1, sys_id)

    mu_overtake = B.gen_bs_predicate(a2, o, lc['s'] - 0.1, eps, sys_id) & \
        B.gen_bs_predicate(-a2, o, - lc['s'] - 0.1, eps, sys_id) & \
        B.gen_bs_predicate(a1, -a1, dx, eps, sys_id) & \
        B.gen_bs_predicate(a3, o, - 1e-1, eps, sys_id) & \
        B.gen_bs_predicate(-a3, o, - 1e-1, eps, sys_id)

    mu_safe = mu_safe_1 & mu_safe_2
    phi_belief = mu_belief.always(0, N)
    phi_neg_belief = neg_mu_belief.eventually(0, N)

    phi = (phi_neg_belief | mu_overtake.always(0, 2).eventually(0, N-2)) & mu_safe.always(0, N)

    return phi


def visualize(agents, xe, xo, mode):

    T = xe.shape[1]
    M = xo.shape[2]

    x_lim = [-5, 215]
    y_lim = [0, 10]
    legend_loc = locs[mode]

    fig = plt.figure(figsize=(5.4, 1.4))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.32)

    plt.plot(x_lim, [lc['l'], lc['l']], linestyle='solid', linewidth=2, color='black')
    plt.plot(x_lim, [lc['m'], lc['m']], linestyle='dashed', linewidth=1, color='black')
    plt.plot(x_lim, [lc['r'], lc['r']], linestyle='solid', linewidth=2, color='black')

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    def _vs(name):

        vl = agents[name].param[1]         # Length of EV
        vw = agents[name].param[1]/4       # Width of EV

        return {'width': vl, 'height': vw}
    
    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')

    # Plot the trajectory of the ego vehicle (EV)
    for i in range(T - 1):
        for j in range(M):
            pov = ax.add_patch(Rectangle(xy=(tf(*xo[j, :3, i], **_vs('oppo'))), **_vs('oppo'), 
                                         angle=xo[j, 2, i]/3.14*180, linewidth=1, linestyle='dotted', edgecolor=(0, 0, 51*4/255), 
                                         fill=True, facecolor=c_oppo(i*0.8/T), zorder=10+i*M+xo[j, 1, i]))

        pev = ax.add_patch(Rectangle(xy=(tf(*xe[:3, i], **_vs('ego'))), **_vs('ego'), 
                                     angle=xe[2, i]/3.14*180, linewidth=1, linestyle='dotted', edgecolor=(51*4/255, 0, 0), 
                                     fill=True, facecolor=c_ego(i*0.8/T), zorder=10+M*T+i*M+xe[1, i]))
        
    i = T - 1

    for j in range(M):
        ax.add_patch(Rectangle(xy=(tf(*xo[j, :3, i], **_vs('oppo'))), **_vs('oppo'), 
                               angle=xo[j, 2, i]/3.14*180, linewidth=1.5, edgecolor=(0, 0, 51/255), 
                               fill=True, facecolor=c_oppo(i*0.8/T), zorder=10+i*M+xo[j, 1, i]))

    ax.add_patch(Rectangle(xy=(tf(*xe[:3, i], **_vs('ego'))), **_vs('ego'), 
                           angle=xe[2, i]/3.14*180, linewidth=1.5, edgecolor=(51/255, 0, 0), 
                           fill=True, facecolor=c_ego(i*0.8/T), zorder=10+M*T+i*M+xe[1, i]))
    
    plt.legend([pev, pov], ['EV', 'OV'], loc=legend_loc, fontsize="8", ncol=2)
    plt.show()


def record(agents, xe, xo, mode, fps=12):

    T = xe.shape[1]
    M = xo.shape[0]
    Ts = agents['ego'].dt
    TT = T * fps * Ts
    dir = 'media/'

    ts = np.arange(0, T)
    tau = np.arange(0, T, 1/(fps*Ts))

    fx = [CubicSpline(ts, xe[i]) for i in range(3)]
    fz = [[CubicSpline(ts, xo[j, i, :]) for j in range(M)] for i in range(3)]

    x_lim = [-5, 215]

    metadata = dict(title='Movie', artist='Zengjie Zhang')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg-full_build\\bin\\ffmpeg.exe'
    
    fig = plt.figure(figsize=(8, 3.85))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.16)

    plt.plot(x_lim, [lc['l'], lc['l']], linestyle='solid', linewidth=2, color='black')
    plt.plot(x_lim, [lc['m'], lc['m']], linestyle='dashed', linewidth=1, color='black')
    plt.plot(x_lim, [lc['r'], lc['r']], linestyle='solid', linewidth=2, color='black')

    plt.ylim([0, 10])
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    def _vs(name):

        vl = agents[name].param[1]         # Length of EV
        vw = agents[name].param[1]/2       # Width of EV

        return {'width': vl, 'height': vw}

    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')

    with writer.saving(fig, dir + 'overtaking_mode_' + str(mode) + '.mp4', 300):

        # Plot the trajectory of the ego vehicle (EV)
        for i, t in enumerate(tau):
            pov = [ax.add_patch(Rectangle(xy=(tf(fz[0][j](t), fz[1][j](t), fz[2][j](t), **_vs('oppo'))), **_vs('oppo'), angle=fz[2][j](t)/3.14*180, linewidth=1.5, linestyle='dotted',
                                    edgecolor=(0, 0, 0.2), fill=True, facecolor=c_oppo(i*0.8/TT), zorder=10+i*M+fz[1][j](t)))
                    for j in range(M)]

            pev = ax.add_patch(Rectangle(xy=(tf(fx[0](t), fx[1](t), fx[2](t), **_vs('ego'))), **_vs('ego'), angle=fx[2](t)/3.14*180, linewidth=1.5, linestyle='dotted',
                                    edgecolor=(0.2, 0, 0), fill=True, facecolor=c_ego(i*0.8/TT), zorder=10+M*TT+i*M+fx[1](t)))
            
            if mode == 0:
                plt.xlim([0.8*i - 6,  0.8*i + 20])
            elif mode == 1:
                plt.xlim([1.0*i - 6,  1.0*i + 20])
            else:
                plt.xlim([1.0*i - 6,  1.0*i + 20])
            writer.grab_frame()
            print('Writing frame ' + str(i) + ' out of ' + str(TT))
            pev.remove()
            for j in range(M):
                pov[j].remove()
    print("---------------------------------------------------------")
    print('Video saved to ' + dir)
    print("---------------------------------------------------------")
        