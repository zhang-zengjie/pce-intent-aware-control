import numpy as np
from config.overtaking.params import initialize
from config.overtaking.functions import visualize, record

# First of first, choose the intent
intent = 2    # Select the intent of certain intentions: 
            # 0 for a switching-lane OV 
            # 1 for a slowing-down OV
            # 2 for a speeding-up OV
N = 15      # Control horizon

# Initialize system and specification
agents, _ = initialize(intent, N)
            # agents: the dictionary of agents
                # agents['ego']: ego vehicle (EV)
                # agents['oppo']: opponent vehicle (OV)

# Load the data of the ego agent
xx = np.load('data/overtaking/x_intent_' + str(intent) + '.npy')

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100     # Number of Monte Carlo runs
zz_s = np.zeros([agents['oppo'].n, N + 1, M])
samples = agents['oppo'].basis.eta.sample([M, ])
for j in range(M):
    agents['oppo'].param = [samples[0, j], samples[1, j], 1]
    agents['oppo'].predict(0, N)
    zz_s[:, :, j] = agents['oppo'].states

if True:
    # Visualize the result
    visualize(xx[:, :N-1], zz_s[:, :N-1, :], intent)

if False:
    # Record the video
    record(xx[:, :10], zz_s[:, :10, :], intent)