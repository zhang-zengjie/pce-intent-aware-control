import numpy as np
import matplotlib.pyplot as plt
import math
from libs.pce_basis import PCEBasis
import chaospy as cp
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import CubicSpline
from libs.commons import tf_anchor as tf

lw = 8                       # The lane width

gray = (102/255, 102/255, 102/255)
light_gray = (230/255, 230/255, 230/255)

# Coefficients of the predicates
o = np.zeros((4, ))
a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])


def get_feedforward(T):

    u1 = np.zeros((2, T))   
    u1[1] -= 0.2                   # OV has deceleration or acceleration behaviors

    u2 = np.zeros((2, T))          # Pedestrians move in constant speeds

    return u1, u2


def get_initial_states():

    e0 = np.array([-lw*2, -lw/2, 0, 3])                  # The ego vehicle (EV) starts with 3 m/s
    o0 = np.array([25, lw/2, math.pi, 2.2])             # The obstacle vehicle (OV) starts with 2.2 m/s
    p0 = np.array([1.2*lw, 1.2*lw, math.pi, 0.5])        # The pedestrians (PD) walk in a average speed 0.5 m/s

    return e0, o0, p0


def get_bases(l, q):

    bias1 = cp.Normal(0, 1e-2)
    intent1 = cp.DiscreteUniform(-1, 1)
    bias2 = cp.Normal(0, 1e-2)
    intent2 = cp.DiscreteUniform(1, 2)

    length1 = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
    eta1 = cp.J(bias1, length1, intent1) # Generate the random variable instance
    B1 = PCEBasis(eta1, q)

    length2 = cp.Uniform(lower=0.5-1e-3, upper=0.5+1e-3)
    eta2 = cp.J(bias2, length2, intent2) # Generate the random variable instance
    B2 = PCEBasis(eta2, q)

    return B1, B2


def approximate(sys, tr, nodes, Q, i):
    oppo_scena = np.zeros([sys['oppo'].n, Q])
    pedes_scena = np.zeros([sys['pedes'].n, Q])
    if i > 0:
        for q in range(0, Q):
            sys['oppo'].param = np.array(nodes['oppo'][:, i, q])
            sys['pedes'].param = np.array(nodes['pedes'][:, i, q])
            sys['oppo'].update_matrices()
            sys['pedes'].update_matrices()
            oppo_scena[:, q] = sys['oppo'].f(tr['oppo'][0, :, i-1], sys['oppo'].useq[:, i-1])
            pedes_scena[:, q] = sys['pedes'].f(tr['pedes'][0, :, i-1], sys['pedes'].useq[:, i-1])
    oppo_std = np.std(oppo_scena, axis=1)
    pedes_std = np.std(pedes_scena, axis=1)
    return oppo_std, pedes_std


def turn_specs(B, N, sys_id):

    reach = B.gen_bs_predicate(a1, o, lw/2 - lw/2, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a1, o, -lw/2 - lw/2, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(a2, o, 3/2*lw, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(a3, o, np.pi/2- 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a3, o, -np.pi/2- 0.1, epsilon=1, name=sys_id) 

    bet_out = B.gen_bs_predicate(a2, o, -lw/2 - 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a2, o, lw/2 - 0.1, epsilon=1, name=sys_id)
    vel_out = B.gen_bs_predicate(a1, o, -1.2*lw, epsilon=1, name=sys_id)
    
    drive_out = vel_out | bet_out

    #phi = r.always(0, 1).eventually(0, N-1) & drive_out.always(0, N) 

    phi = reach.eventually(0, N) & drive_out.always(0, N)

    return phi

def safety_specs(B, N, sys_id, dist=4, eps=0.05):
    
    mu_safe = B.gen_bs_predicate(a1, -a1, dist, epsilon=eps, name=sys_id) | \
                B.gen_bs_predicate(-a1, a1, dist, epsilon=eps, name=sys_id) | \
                B.gen_bs_predicate(a2, -a2, dist, epsilon=eps, name=sys_id) | \
                B.gen_bs_predicate(-a2, a2, dist, epsilon=eps, name=sys_id)
    
    phi = mu_safe.always(0, N)

    return phi

def safety_specs_multi_modal(B, N, sys_id, std, dist=4, eps=0.05):
    
    offset_x = math.sqrt((1 - eps) / eps) * std[0] + dist
    offset_y = math.sqrt((1 - eps) / eps) * std[1] + dist

    mu_safe_1 = B.gen_bs_predicate(a1, -a1, offset_x, epsilon=1, name=sys_id) & B.gen_bs_predicate(a1, -a1, -offset_x, epsilon=1, name=sys_id)
    mu_safe_2 = B.gen_bs_predicate(-a1, a1, offset_x, epsilon=1, name=sys_id) & B.gen_bs_predicate(-a1, a1, -offset_x, epsilon=1, name=sys_id)
    mu_safe_3 = B.gen_bs_predicate(a2, -a2, offset_y, epsilon=1, name=sys_id) & B.gen_bs_predicate(a2, -a2, -offset_y, epsilon=1, name=sys_id)
    mu_safe_4 = B.gen_bs_predicate(-a2, a2, offset_y, epsilon=1, name=sys_id) & B.gen_bs_predicate(-a2, a2, -offset_y, epsilon=1, name=sys_id)
    
    mu_safe = mu_safe_1 | mu_safe_2 | mu_safe_3 | mu_safe_4

    phi = mu_safe.always(0, N)

    return phi


def get_specs(sys, N, mode):
    phi_ego = turn_specs(sys['oppo'].basis, N, 'ego')
    if mode == 0:
        phi = phi_ego
    else:
        phi_oppo = safety_specs(sys['oppo'].basis, N, dist=1, sys_id='oppo')
        phi_pedes = safety_specs(sys['pedes'].basis, N, dist=1, sys_id='pedes')
        phi = phi_ego & phi_oppo & phi_pedes
    return phi


def model_checking(x, z, spec, k):

    L = (1 + z.shape[0]) * z.shape[1]
    xx = np.zeros([L, z.shape[2]])

    for i in range(z.shape[2]):
        xx[:z.shape[1], i] = x[:, i]
        xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

    rho = spec.robustness(xx, k)

    return rho


def visualize(agents, xe, xo, xp, cursor):

    T = xe.shape[1]
    M = xo.shape[0]

    x_lim = [-3*lw, 3*lw]
    y_lim = [-3*lw, 3*lw]

    fig = plt.figure(figsize=(3.5, 3.3))
    ax = plt.axes()

    # Draw the environment
    for i in [-1, 1]:

        for r in np.arange(-0.9, 1, 0.1):
            plt.plot([r*lw, r*lw], [1.1*i*lw, 1.4*i*lw], color=gray, linewidth=2, zorder=-20)
            plt.plot([1.1*i*lw, 1.4*i*lw], [r*lw, r*lw], color=gray, linewidth=2, zorder=-20)

        plt.plot([0, 0], [1.5*i*lw, 3*i*lw], color='black', linewidth=1, zorder=-20)
        plt.plot([1.5*i*lw, 3*i*lw], [0, 0], color='black', linewidth=1, zorder=-20)

        for j in [-1, 1]:
            plt.plot([i*lw, i*lw], [j*lw, 3*j*lw], color='black', linewidth=2, zorder=-20)
            plt.plot([j*lw, 3*j*lw], [i*lw, i*lw], color='black', linewidth=2, zorder=-20)

            plt.plot([0.5*i*lw, 0.5*i*lw], [1.5*j*lw, 3*j*lw], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)
            plt.plot([1.5*j*lw, 3*j*lw], [0.5*i*lw, 0.5*i*lw], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)

    def _vs(name):

        vl = agents[name].param[1]         # Length of EV
        vw = agents[name].param[1]/2       # Width of EV

        return {'width': vl, 'height': vw}
    
    # Plot the trajectory of the ego vehicle (EV)

    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')
    c_pedes = plt.get_cmap('YlOrBr')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('x position (m)', fontsize="12")
    plt.ylabel('y position (m)', fontsize="12")
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.13)
    fig.tight_layout()

    # Plot the trajectory of the ego vehicle (EV)
    for i in range(T):
        ax.add_patch(Rectangle(xy=tf(*xe[:3, i], **_vs('ego')), angle=xe[2, i]*180/np.pi, **_vs('ego'), 
                               linewidth=1.5, linestyle=':', fill=True, edgecolor='red', facecolor=c_ego((i/T)**1), zorder=0))
        
    pev = ax.add_patch(Rectangle(xy=tf(*xe[:3, cursor], **_vs('ego')), angle=xe[2, cursor]*180/np.pi, **_vs('ego'), 
                               linewidth=1.5, fill=True, edgecolor='black', facecolor=c_ego((cursor/T)**1), zorder=0))

    # Plot the sampled trajectories of the obstacle vehicle (OV) 
    for j in range(M):
        
        pov = ax.add_patch(Rectangle(xy=tf(*xo[j, :3, cursor], **_vs('ego')), angle=xo[j, 2, cursor]*180/np.pi, **_vs('ego'), 
                                linewidth=1, linestyle='--', fill=True, edgecolor='black', facecolor=c_oppo((cursor/T)**1), zorder=20-xo[j, 0, cursor]))

        ppd = ax.add_patch(Circle(xy=tuple(xp[j, :2, cursor]), radius=_vs('pedes')['width'], linewidth=1.5, linestyle='--', fill=True, 
                                edgecolor='black', facecolor=c_pedes((cursor/T)**1), zorder=20-xp[j, 0, cursor]))

    plt.legend([pev, pov, ppd], ['Ego vehicle', 'Opponent vehicle', 'Pedestrian'], loc=(0.03, 0.03), fontsize="10", ncol=1)
    plt.show()


def record(agents, xe, xo, xp, mode, fps=12):

    T = xe.shape[1]
    M = xo.shape[0]
    Ts = agents['ego'].dt
    TT = T * fps * Ts
    dir = 'media/'

    x_lim = [-3*lw, 3*lw]
    y_lim = [-3*lw, 3*lw]

    fig = plt.figure(figsize=(6, 5.7))
    ax = plt.axes()
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg-full_build\\bin\\ffmpeg.exe'

    ts = np.arange(0, T)
    tau = np.arange(0, T, 1/(fps*Ts))

    fe = [CubicSpline(ts, xe[i]) for i in range(3)]
    fo = [[CubicSpline(ts, xo[j, i, :]) for j in range(M)] for i in range(3)]
    fp = [[CubicSpline(ts, xp[j, i, :]) for j in range(M)] for i in range(3)]

    # Draw the environment

    for i in [-1, 1]:

        for r in np.arange(-0.9, 1, 0.1):
            plt.plot([r*lw, r*lw], [1.1*i*lw, 1.4*i*lw], color=gray, linewidth=2, zorder=-20)
            plt.plot([1.1*i*lw, 1.4*i*lw], [r*lw, r*lw], color=gray, linewidth=2, zorder=-20)

        plt.plot([0, 0], [1.5*i*lw, 3*i*lw], color='black', linewidth=1, zorder=-20)
        plt.plot([1.5*i*lw, 3*i*lw], [0, 0], color='black', linewidth=1, zorder=-20)

        for j in [-1, 1]:
            plt.plot([i*lw, i*lw], [j*lw, 3*j*lw], color='black', linewidth=2, zorder=-20)
            plt.plot([j*lw, 3*j*lw], [i*lw, i*lw], color='black', linewidth=2, zorder=-20)

            plt.plot([0.5*i*lw, 0.5*i*lw], [1.5*j*lw, 3*j*lw], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)
            plt.plot([1.5*j*lw, 3*j*lw], [0.5*i*lw, 0.5*i*lw], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)

    def _vs(name):

        vl = agents[name].param[1]         # Length of EV
        vw = agents[name].param[1]/2       # Width of EV

        return {'width': vl, 'height': vw}
    # Plot the trajectory of the ego vehicle (EV)

    c_ego = plt.get_cmap('Reds')
    c_oppo = plt.get_cmap('Blues')
    c_pedes = plt.get_cmap('YlOrBr')
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('x position (m)', fontsize="12")
    plt.ylabel('y position (m)', fontsize="12")
    # plt.legend([pev, pov, ppd], ['Ego vehicle', 'Opponent vehicle', 'Pedestrian'], loc=(0.03, 0.03), fontsize="10", ncol=1)
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.13)
    fig.tight_layout()

    metadata = dict(title='Movie', artist='Zengjie Zhang')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Plot the sampled trajectories of the obstacle vehicle (OV) 
    with writer.saving(fig, dir + 'intersection_scene_' + str(mode) + '.mp4', 300):

        for i, t in enumerate(tau):
        
            pev = ax.add_patch(Rectangle(xy=tf(fe[0](t), fe[1](t), fe[2](t), **_vs('ego')), angle=fe[2](t)*180/np.pi, 
                               **_vs('ego'), linewidth=1.5, linestyle=':', fill=True,
                               edgecolor='red', facecolor=c_ego((i/TT)**1), zorder=50))

            pov = [ax.add_patch(Rectangle(xy=tf(fo[0][j](t), fo[1][j](t), fo[2][j](t), **_vs('ego')), angle=fo[2][j](t)*180/np.pi, 
                                    **_vs('oppo'), linewidth=1, linestyle='--', fill=True, 
                                    edgecolor='black', facecolor=c_oppo((i/TT)**1), zorder=20-fo[0][j](t)))
                    for j in range(M)]
            
            ppd = [ax.add_patch(Circle(xy=tuple([fp[0][j](t), fp[1][j](t), fp[2][j](t)]), radius=0.5, linewidth=1.5, linestyle='--', fill=True, 
                                    edgecolor='black', facecolor=c_pedes((i/TT)**1), zorder=20-fp[0][j](t)))
                    for j in range(M)]
            
            writer.grab_frame()
            print('Writing frame ' + str(i) + ' out of ' + str(TT))
            pev.remove()
            for j in range(M):
                pov[j].remove()
                ppd[j].remove() 

    print("---------------------------------------------------------")
    print('Video saved to ' + dir)
    print("---------------------------------------------------------")
