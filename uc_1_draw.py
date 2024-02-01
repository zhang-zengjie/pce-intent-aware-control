import numpy as np
from libs.micp_pce_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from config.uc_1_config import gen_pce_specs, lanes, visualize
from libs.commons import model_checking
from libs.pce_basis import PCEBasis
import chaospy as cp

x = np.load('x_switch_lane.npy')

visualize(x, [200, 400], oppo)
