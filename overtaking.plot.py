import numpy as np
from config.overtaking.params import initialize
from config.overtaking.functions import visualize, record

# First of first, choose the mode
mode = 2    # Select the mode of certain intentions: 
            # 0 for a switching-lane OV 
            # 1 for a slowing-down OV
            # 2 for a speeding-up OV
N = 15      # Control horizon

# Initialize system and specification
sys, _ = initialize(mode, N)
            # sys: the dictionary of agents

# Load the data of the ego agent
xx = np.load('data/overtaking/x_mode_' + str(mode) + '.npy')

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100
zz_s = np.zeros([sys['oppo'].n, N + 1, M])
samples = sys['oppo'].basis.eta.sample([M, ])
for j in range(M):
    sys['oppo'].param = [samples[0, j], samples[1, j], 1]
    sys['oppo'].predict(0, N)
    zz_s[:, :, j] = sys['oppo'].states

if True:
    # Visualize the result
    visualize(xx[:, :N-1], zz_s[:, :N-1, :], mode)

if False:
    # Record the video
    record(xx[:, :10], zz_s[:, :10, :], mode)