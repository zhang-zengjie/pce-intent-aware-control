import numpy as np
from libs.pce_micp_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from commons import gen_pce_specs, lanes
from commons import visualize

import chaospy as cp


ASSUMED_INPUT = "speed_up"
base_sampling_time = 0.5    # The baseline value of sampling time delta_t
base_length = 4             # The baseline value of the vehicle length
q = 2                       # The polynomial order
N = 30                      # The control horizon

np.random.seed(7)

if ASSUMED_INPUT == "switch_lane":  
    sigma = 0.5
    v = np.load('v.npy')
elif ASSUMED_INPUT == "slow_down":
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, -2, N)
    v = np.array([gamma, a])
elif ASSUMED_INPUT == "speed_up":
    sigma = 0.1
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 2, N)
    v = np.array([gamma, a])
else: # "big_variance"
    sigma = 0.5
    gamma = np.linspace(0, 0, N)
    a = np.linspace(0, 0, N)
    v = np.array([gamma, a])


# The assumed control input of the obstacle vehicle (OV)


delta = cp.Trunc(cp.Normal(0, sigma), lower=-sigma, upper=sigma)
length = cp.Uniform(lower=base_length - 1e-2, upper=base_length + 1e-2)
eta = cp.J(delta, length) # Generate the random variable instance

x0 = np.array([0, lanes['fast'], 0, 25])            # Initial position of the ego vehicle (EV)
z0 = np.array([50, lanes['slow'], 0, 25])           # Initial position of the obstacle vehicle (OV)

# Generate the PCE instance and the specification
B, phi = gen_pce_specs(q, N, eta)

sys1 = BicycleModel(x0, [0, base_length], base_sampling_time)                  # Dynamic model of the ego vehicle (EV)
sys2 = BicycleModel(z0, [0, base_length], base_sampling_time, B, pce=True)     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
solver = PCEMICPSolver(phi, sys1, sys2, v, N, robustness_cost=False)

# Adding input constraints (not necessary if input is in the cost function)
#u_min = np.array([[-0.5, -50]]).T
#u_max = np.array([[0.5, 50]]).T
#solver.AddControlBounds(u_min, u_max)

# Adding input to the cost function
Q = np.zeros([sys1.n, sys1.n])
# Q = np.diag([0, 10, 10, 0])
R = np.array([[1e4, 0], [0, 1e-4]])
ref = np.array([0, lanes['fast'], 0, 0])
solver.AddQuadraticCost(Q, R, ref)

# Solve the problem
x, z, u, _, _ = solver.Solve()
L = (1 + z.shape[0]) * z.shape[1]
xx = np.zeros([L, z.shape[2]])

# u[0] = -u[0]
# np.save('v.npy', u)

for i in range(z.shape[2]):
    xx[:z.shape[1], i] = x[:, i]
    xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

# Visualize the results
# mu_belief.robustness(xx, 1)
visualize(x, z0, v, B, sys2)
