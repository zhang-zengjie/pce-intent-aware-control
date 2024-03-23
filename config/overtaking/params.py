import numpy as np
from config.overtaking.functions import gen_bases, get_intention, gen_pce_specs, lanes
from libs.bicycle_model import BicycleModel


def initialize(mode, N):

    dt = 1                                       # The discrete sampling time Delta_t
    l = 4                                        # The baseline value of the vehicle length
    R = np.array([[1e4, 0], [0, 1e-6]])          # Control cost
    np.random.seed(7)
    
    v = get_intention(N, mode)                   # The certain intention of the obstacle vehicle (OV)
    B = gen_bases(l)                             # The chaos basis object
    
    v0 = 10          
    e0 = np.array([0, lanes['fast'], 0, v0*1.2]) # Initial position of the ego vehicle (EV)
    o0 = np.array([2*v0, lanes['slow'], 0, v0])  # Initial position of the obstacle vehicle (OV)
    
    sys = {'ego': BicycleModel(dt, x0=e0, param=[0, l, 1], N=N, useq=np.zeros(v.shape), R=R, name='ego'),     # Dynamic model of the ego vehicle (EV) 
           'oppo': BicycleModel(dt, x0=o0, param=[0, l, 1], N=N, useq=v, basis=B, pce=True, name='oppo')     # Dynamic model of the obstacle vehicle (OV)
           }
    
    phi = gen_pce_specs(B, N, v0*1.2, 'oppo')    # Specification

    return sys, phi