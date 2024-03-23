import numpy as np
from config.intersection.functions import gen_bases, get_intentions, get_initials, get_spec
from config.intersection.functions import veh_len as l
from libs.bicycle_model import BicycleModel


Ts = 0.5          # The baseline value of sampling time delta_t
N = 35              # The control horizon
M = 100               # Runs
R = np.array([[1, 0], [0, 1]])
Q = 25              # Scenario numbers

mode = 1    # Select simulation mode: 
            # 0 for no_reaction 
            # 1 for reaction with proposed method

np.random.seed(7)

v1, v2 = get_intentions(N)          # The assumed control mode of the opponent vehicle (OV)
B1, B2 = gen_bases()                # Generate PCE bases
e0, o0, p0 = get_initials()         # Get initial conditions

sys = {'ego': BicycleModel(Ts, x0=e0, param=[0, l, 1], N=N, useq=np.zeros(v1.shape), R=R, name='ego'),                                     # Dynamic model of the ego vehicle (EV)
       'oppo': BicycleModel(Ts, x0=o0, param=[0, l, 1], N=N, useq=v1, basis=B1, pce=True, name='oppo'),      # Dynamic model of the opponent vehicle (OV)
       'pedes': BicycleModel(Ts, x0=p0, param=[0, l, 1], N=N, useq=v2, basis=B2, pce=True, name='pedes')}    # Dynamic model of the pedestrian (PD)

phi = get_spec(sys, N, mode)