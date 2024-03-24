import numpy as np
from config.intersection.functions import get_bases, get_feedforward, get_initial_states, get_specs
from libs.bicycle_model import BicycleModel


def initialize(mode, N):

    dt = 0.5                       # Baseline value of sampling time delta_t    
    l = 3.6                          # Baseline value of the vehicle length
    q = 2                          # Polynomial order
    R = np.array([[1, 0], [0, 1]]) # Control cost

    np.random.seed(7)

    v1, v2 = get_feedforward(N)          # The assumed control mode of the opponent vehicle (OV)
    B1, B2 = get_bases(l, q)                # Generate PCE bases
    e0, o0, p0 = get_initial_states()         # Get initial conditions

    sys = {'ego': BicycleModel(dt, x0=e0, param=[0, l, 1], N=N, useq=np.zeros(v1.shape), R=R, name='ego'),                                     # Dynamic model of the ego vehicle (EV)
           'oppo': BicycleModel(dt, x0=o0, param=[0, l, 1], N=N, useq=v1, basis=B1, pce=True, name='oppo'),      # Dynamic model of the opponent vehicle (OV)
           'pedes': BicycleModel(dt, x0=p0, param=[0, 0.5, 1], N=N, useq=v2, basis=B2, pce=True, name='pedes')}    # Dynamic model of the pedestrian (PD)

    phi = get_specs(sys, N, mode)

    return sys, phi