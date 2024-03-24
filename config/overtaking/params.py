import numpy as np
from config.overtaking.functions import get_bases, get_feedforward, get_initial_states, get_specs
from libs.bicycle_model import BicycleModel


def initialize(mode, N):

    dt = 1                                       # The discrete sampling time Delta_t
    l = 4                                        # The baseline value of the vehicle length
    q = 2                               # The polynomial order
    R = np.array([[1e4, 0], [0, 1e-6]])          # Control cost

    np.random.seed(7)
    
    v = get_feedforward(N, mode)                   # The certain intention of the obstacle vehicle (OV)
    B = get_bases(l, q)                             # The chaos basis object
    e0, o0 = get_initial_states()
    
    sys = {'ego': BicycleModel(dt, x0=e0, param=[0, l, 1], N=N, useq=np.zeros(v.shape), R=R, name='ego'),     # Dynamic model of the ego vehicle (EV) 
           'oppo': BicycleModel(dt, x0=o0, param=[0, l, 1], N=N, useq=v, basis=B, pce=True, name='oppo')     # Dynamic model of the obstacle vehicle (OV)
           }
    
    phi = get_specs(B, N, 'oppo')    # Specification

    return sys, phi