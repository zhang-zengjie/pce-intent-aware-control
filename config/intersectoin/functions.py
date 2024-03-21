import numpy as np
import matplotlib.pyplot as plt
from libs.pce_basis import PCEBasis
import math
from matplotlib.patches import Rectangle, Circle
from libs.pce_basis import PCEBasis
import chaospy as cp


l = 8                       # The lane width
q = 2                       # The polynomial order
veh_width = 1.8
veh_len = 3.6

gray = (102/255, 102/255, 102/255)
light_gray = (230/255, 230/255, 230/255)

# Coefficients of the predicates
o = np.zeros((4, ))
a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])

def get_intentions(T):

    gamma1 = np.linspace(0, 0, T)
    a1 = np.linspace(0, -0.5, T)
    u1 = np.array([gamma1, a1])

    gamma2 = np.linspace(0, 0, T)
    a2 = np.linspace(0.02, 0, T)
    u2 = np.array([gamma2, a2])

    return u1, u2

def get_initials():

    e0 = np.array([-l*2, -l/2, 0, 2])              # Initial position of the ego vehicle (EV)
    o0 = np.array([105, l/2, math.pi, 8])             # Initial position of the obstacle vehicle (OV)
    p0 = np.array([1.2*l, 1.2*l, math.pi, 0])        # Initial position of the pedestrian (PD)

    return e0, o0, p0


def gen_bases():

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

    reach = B.gen_bs_predicate(a1, o, l/2 - 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a1, o, -l/2 - 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(a2, o, 3/2*l, epsilon=1, name=sys_id)

    drive_in = B.gen_bs_predicate(a3, o, np.pi/2- 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a3, o, -np.pi/2- 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(a1, o, l/2 - 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a1, o, -l/2 - 0.1, epsilon=1, name=sys_id)

    bet_out = B.gen_bs_predicate(a2, o, -l/2 - 0.1, epsilon=1, name=sys_id) & \
        B.gen_bs_predicate(-a2, o, l/2 - 0.1, epsilon=1, name=sys_id)
    vel_out = B.gen_bs_predicate(a1, o, -1.2*l, epsilon=1, name=sys_id)
    
    drive_out = vel_out | bet_out

    if N > 4:
        phi = reach.always(0, 3).eventually(0, N-3) & drive_out.always(0, N) 
    elif N > 2:
        phi = drive_in.always(0, 2).eventually(0, N-2)
    else:
        phi = drive_in.always(0, N)

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


def get_spec(sys, tr, samples, Q, N, i, mode):
    phi_ego = turn_specs(sys['oppo'].basis, N-i, 'ego')
    if mode == 0:
        phi = phi_ego
    elif mode == 2:
        oppo_std, pedes_std = approximate(sys, tr, samples, Q, i)
        phi_oppo = safety_specs_multi_modal(sys['oppo'].basis, N-i, std=oppo_std, dist=8, sys_id='oppo')
        phi_pedes = safety_specs_multi_modal(sys['pedes'].basis, N-i, std=pedes_std, dist=2, sys_id='pedes')
        phi = phi_ego & phi_oppo & phi_pedes
    else:
        phi_oppo = safety_specs(sys['oppo'].basis, N-i, dist=4, sys_id='oppo')
        phi_pedes = safety_specs(sys['pedes'].basis, N-i, dist=2, sys_id='pedes')
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


def visualize(tr_ego, tr_oppo, tr_pedes, cursor):

    fig = plt.figure(figsize=(3.5, 3.3))
    ax = plt.axes()
    
    x_lim = [-3*l, 3*l]
    y_lim = [-3*l, 3*l]

    T = tr_ego.shape[1]
    M = tr_oppo.shape[0]

    gray = (102/255, 102/255, 102/255)
    light_gray = (230/255, 230/255, 230/255)

    # Draw the environment

    for i in [-1, 1]:

        for r in np.arange(-0.9, 1, 0.1):
            plt.plot([r*l, r*l], [1.1*i*l, 1.4*i*l], color=gray, linewidth=2, zorder=-20)
            plt.plot([1.1*i*l, 1.4*i*l], [r*l, r*l], color=gray, linewidth=2, zorder=-20)

        plt.plot([0, 0], [1.5*i*l, 3*i*l], color='black', linewidth=1, zorder=-20)
        plt.plot([1.5*i*l, 3*i*l], [0, 0], color='black', linewidth=1, zorder=-20)

        for j in [-1, 1]:
            plt.plot([i*l, i*l], [j*l, 3*j*l], color='black', linewidth=2, zorder=-20)
            plt.plot([j*l, 3*j*l], [i*l, i*l], color='black', linewidth=2, zorder=-20)

            plt.plot([0.5*i*l, 0.5*i*l], [1.5*j*l, 3*j*l], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)
            plt.plot([1.5*j*l, 3*j*l], [0.5*i*l, 0.5*i*l], color=light_gray, linewidth=1, linestyle='dotted', zorder=-20)

    # Plot the trajectory of the ego vehicle (EV)
            
    def tf_anchor(x, y, theta):
        xr = x - math.cos(theta) * veh_len + math.sin(theta) * veh_width/2
        yr = y - math.sin(theta) * veh_len - math.cos(theta) * veh_width/2
        return (xr, yr)

    c_ego = plt.get_cmap('Reds')
    for i in range(0, T):
        ax.add_patch(Rectangle(xy=tf_anchor(*tr_ego[:3, i]), angle=tr_ego[2, i]*180/np.pi, 
                               width=veh_len, height=veh_width, linewidth=1.5, linestyle=':', fill=True,
                               edgecolor='red', facecolor=c_ego((i/T)**4), zorder=50))
        
    pev = ax.add_patch(Rectangle(xy=tf_anchor(*tr_ego[:3, cursor]), angle=tr_ego[2, cursor]*180/np.pi, 
                               width=veh_len, height=veh_width, linewidth=1.5, fill=True,
                               edgecolor='black', facecolor=c_ego((cursor/T)**4), zorder=50))
        
    c_oppo = plt.get_cmap('Blues')
    c_pedes = plt.get_cmap('YlOrBr')

    # Plot the sampled trajectories of the obstacle vehicle (OV) 

    for j in range(M):
        
        pov = ax.add_patch(Rectangle(xy=tf_anchor(*tr_oppo[j, :3, cursor]), angle=tr_oppo[j, 2, cursor]*180/np.pi, 
                                width=4, height=2, linewidth=1, linestyle='--', fill=True, 
                                edgecolor='black', facecolor=c_oppo((cursor/T)**4), zorder=20-tr_oppo[j, 0, cursor]))

        ppd = ax.add_patch(Circle(xy=tuple(tr_pedes[j, :2, cursor]), radius=0.5, linewidth=1.5, linestyle='--', fill=True, 
                                edgecolor='black', facecolor=c_pedes((cursor/T)**4), zorder=20-tr_pedes[j, 0, cursor]))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('x position (m)', fontsize="12")
    plt.ylabel('y position (m)', fontsize="12")
    plt.legend([pev, pov, ppd], ['Ego vehicle', 'Opponent vehicle', 'Pedestrian'], loc=(0.03, 0.03), fontsize="10", ncol=1)
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.13)
    fig.tight_layout()

    plt.show()
