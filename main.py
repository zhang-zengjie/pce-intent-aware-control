import numpy as np
from libs.pce_micp_solver import PCEMICPSolver
from libs.bicycle_model import BicycleModel
from commons import gen_pce_specs, lanes
from commons import visualize


base_sampling_time = 0.5    # The baseline value of sampling time delta_t
base_length = 4             # The baseline value of the vehicle length
q = 2                       # The polynomial order
N = 30                      # The control horizon

# Generate the PCE instance and the specification
B, phi = gen_pce_specs(base_sampling_time, base_length, q, N)

# The assumed control input of the obstacle vehicle (OV)
gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
v = np.array([gamma, a])

x0 = np.array([0, lanes['fast'], 0, 25])            # Initial position of the ego vehicle (EV)
z0 = np.array([50, lanes['slow'], 0, 25])           # Initial position of the obstacle vehicle (OV)

sys1 = BicycleModel(x0, [base_sampling_time, base_length])                  # Dynamic model of the ego vehicle (EV)
sys2 = BicycleModel(z0, [base_sampling_time, base_length], B, pce=True)     # Dynamic model of the obstacle vehicle (OV)

# Initialize the solver
solver = PCEMICPSolver(phi, sys1, sys2, v, N, robustness_cost=True)

# Adding input constraints (not necessary if input is in the cost function)
# u_min = np.array([[-0.5, 0]]).T
# u_max = np.array([[0.5, 10]]).T
# solver.AddControlBounds(u_min, u_max)

# Adding input to the cost function
Q = np.zeros([sys1.n, sys1.n])
R = np.array([[0.5, 0], [0, 0.01]])
solver.AddQuadraticCost(Q, R)

# Solve the problem
x, z, u, _, _ = solver.Solve()

# Visualize the results
visualize(x, z0, v, B, sys2)