import numpy as np
import matplotlib.pyplot as plt
from libs.pce_basis import PCEBasis
from stlpy.STL import LinearPredicate
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
from matplotlib.patches import Rectangle, Circle
from libs.pce_basis import PCEBasis
import chaospy as cp


l = 8 # The lane width
q = 2                       # The polynomial order
veh_width = 1.8
veh_len = 3.6

gray = (102/255, 102/255, 102/255)
light_gray = (230/255, 230/255, 230/255)

x_lim = [-30*l, 30*l]
y_lim = [-3*l, 3*l]
z_lim = [-3*l, 3*l]

# Coefficients of the predicates
o = np.zeros((4, ))
a1 = np.array([1, 0, 0, 0])
a2 = np.array([0, 1, 0, 0])
a3 = np.array([0, 0, 1, 0])
a4 = np.array([0, 0, 0, 1])

M = 64

def get_intentions(T):

    gamma1 = np.linspace(0, 0, T)
    a1 = np.linspace(0, -1.2, T)
    u1 = np.array([gamma1, a1])

    gamma2 = np.linspace(0, 0, T)
    a2 = np.linspace(0, 0.5, T)
    u2 = np.array([gamma2, a2])

    return u1, u2


def gen_bases(mode):
    
    if mode == 2:
        bias1 = cp.Normal(0, math.sqrt(2/3))
        intent1 = cp.Normal(0, 1e-3)
        bias2 = cp.Normal(0, 0.5)
        intent2 = cp.Normal(0.5, 1e-3)
    else:
        bias1 = cp.Normal(0, 1e-2)
        intent1 = cp.DiscreteUniform(-1, 1)
        bias2 = cp.Normal(0, 0.01)
        intent2 = cp.DiscreteUniform(0, 1)

    length1 = cp.Uniform(lower=l-1e-2, upper=l+1e-2)
    eta1 = cp.J(bias1, length1, intent1) # Generate the random variable instance
    B1 = PCEBasis(eta1, q)

    length2 = cp.Uniform(lower=0.5-1e-3, upper=0.5+1e-3)
    eta2 = cp.J(bias2, length2, intent2) # Generate the random variable instance
    B2 = PCEBasis(eta2, q)

    return B1, B2

def turn_specs(B, N, sys_id):

    reach = B.expectation_formula(a1, o, l/2 - 0.1, name=sys_id) & \
        B.expectation_formula(-a1, o, -l/2 - 0.1, name=sys_id) & \
        B.expectation_formula(a2, o, 3/2*l, name=sys_id)

    drive_in = B.expectation_formula(a1, o, l/2 - 0.5, name=sys_id) & \
        B.expectation_formula(-a1, o, -l/2 - 0.5, name=sys_id)

    bet_out = B.expectation_formula(a2, o, -l/2 - 0.1, name=sys_id) & \
        B.expectation_formula(-a2, o, l/2 - 0.1, name=sys_id)
    vel_out = B.expectation_formula(a1, o, -1.2*l, name=sys_id)
    
    drive_out = vel_out | bet_out

    if N > 3:
        phi = reach.always(0, 3).eventually(0, N-3) & drive_out.always(0, N) 
    else:
        phi = drive_in.always(0, N)

    return phi

def safety_specs(B, N, sys_id, dist=2, eps=0.1):
    
    mu_safe = B.probability_formula(a1, -a1, dist, eps, name=sys_id) | \
        B.probability_formula(-a1, a1, dist, eps, name=sys_id) | \
        B.probability_formula(a2, -a2, dist, eps, name=sys_id) | \
        B.probability_formula(-a2, a2, dist, eps, name=sys_id)
    
    phi = mu_safe.always(0, N)

    return phi


def model_checking(x, z, spec, k):

    L = (1 + z.shape[0]) * z.shape[1]
    xx = np.zeros([L, z.shape[2]])

    for i in range(z.shape[2]):
        xx[:z.shape[1], i] = x[:, i]
        xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

    rho = spec.robustness(xx, k)

    return rho


def visualize(sys, tr, draw_real_oppo_traj=True):

    fig = plt.figure(figsize=(30, 3))
    ax = plt.axes()
    T = tr['ego'].shape[1]
    cursor = 24

    gray = (102/255, 102/255, 102/255)
    light_gray = (230/255, 230/255, 230/255)
    # Draw the environment
    for i in [-1, 1]:

        for r in np.arange(-0.9, 1, 0.1):
            plt.plot([r*l, r*l], [1.1*i*l, 1.4*i*l], color=gray, linewidth=2)
            plt.plot([1.1*i*l, 1.4*i*l], [r*l, r*l], color=gray, linewidth=2)

        plt.plot([0, 0], [1.5*i*l, 3*i*l], color='black', linewidth=1)
        plt.plot([1.5*i*l, 3*i*l], [0, 0], color='black', linewidth=1)

        for j in [-1, 1]:
            plt.plot([i*l, i*l], [j*l, 3*j*l], color='black', linewidth=2)
            plt.plot([j*l, 3*j*l], [i*l, i*l], color='black', linewidth=2)

            plt.plot([0.5*i*l, 0.5*i*l], [1.5*j*l, 3*j*l], color=light_gray, linewidth=1, linestyle='dotted')
            plt.plot([1.5*j*l, 3*j*l], [0.5*i*l, 0.5*i*l], color=light_gray, linewidth=1, linestyle='dotted')

    # Plot the trajectory of the ego vehicle (EV)
    tr1, = plt.plot(tr['ego'][0, :], tr['ego'][1, :], linestyle='solid', linewidth=2, color='red')

    def tf_anchor(x, y, theta):
        xr = x - math.cos(theta) * veh_len + math.sin(theta) * veh_width/2
        yr = y - math.sin(theta) * veh_len - math.cos(theta) * veh_width/2
        return (xr, yr)

    c_ego = plt.get_cmap('Reds')
    for i in range(0, T):
        ax.add_patch(Rectangle(xy=tf_anchor(*tr['ego'][:3, i]), angle=tr['ego'][2, i]*180/np.pi, 
                               width=veh_len, height=veh_width, linewidth=1, fill=True,
                               edgecolor=sys['ego'].color, facecolor=c_ego(i/T), zorder=10))
        
    c_oppo = plt.get_cmap('Blues')
    c_pedes = plt.get_cmap('YlOrBr')
    if draw_real_oppo_traj:

        for i in range(0, T):

            ax.add_patch(Rectangle(xy=tf_anchor(*tr['oppo'][:3, i]), angle=tr['oppo'][2, i]*180/np.pi, 
                            width=veh_len, height=veh_width, linewidth=1, fill=True,
                            edgecolor=sys['oppo'].color, facecolor=c_oppo((i/T)**4), zorder=10))
            ax.add_patch(Circle(xy=tuple(tr['pedes'][:2, i]), radius=0.5, linewidth=1, fill=True,
                            edgecolor=sys['pedes'].color, facecolor=c_pedes((i/T)**4), zorder=10))

    else:    

        nodes = sys.basis.eta.sample([M, len(sys)])

        # Generate the sampled trajectories of the obstacle vehicle (OV) 
        
        mc_oppo = np.zeros([M, 4, T])
        mc_pedes = np.zeros([M, 4, T])

        for j in range(M):
            
            sys['oppo'].param = nodes[:, j, 1]
            sys['oppo'].update_matrices()
            mc_oppo[j] = sys['oppo'].predict_lin(T - 1)
            ax.add_patch(Rectangle(xy=tf_anchor(*mc_oppo[j, :3, cursor]),
                                   width=4, height=2, linewidth=1, 
                                   edgecolor=sys['oppo'].color, facecolor='white', fill=False, zorder=10))

            sys['pedes'].param = nodes[:, j, 2]
            sys['pedes'].update_matrices()
            mc_pedes[j] = sys['pedes'].predict_lin(T - 1)
            ax.add_patch(Rectangle(xy=tf_anchor(*mc_pedes[j, :3, cursor]),
                                   width=4, height=2, linewidth=1, 
                                   edgecolor=sys['pedes'].color, facecolor='white', fill=False, zorder=10))

    ax.add_patch(Rectangle(xy=tf_anchor(*tr['ego'][:3, cursor]), angle=tr['ego'][2, cursor]*180/np.pi, 
                               width=veh_len, height=veh_width, linewidth=2, fill=True,
                               edgecolor='black', facecolor=c_ego(cursor/T), zorder=10))
    
    ax.add_patch(Rectangle(xy=tf_anchor(*tr['oppo'][:3, cursor]), angle=tr['oppo'][2, cursor]*180/np.pi, 
                            width=veh_len, height=veh_width, linewidth=2, fill=True,
                            edgecolor='black', facecolor=c_oppo((cursor/T)**4), zorder=10))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    fig.tight_layout()

    plt.show()
