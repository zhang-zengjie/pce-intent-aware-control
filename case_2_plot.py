import numpy as np
from config.intersection.params import initialize
from config.intersection.functions import visualize, record

# First of first, choose the scenario
scene = 0    # Select simulation scenario: 
        # 0 for no_reaction 
        # 1 for reaction with proposed method
N = 35

# Initialize system and specification
agents, phi = initialize(scene, N)
            # agents: the dictionary of agents
            # phi: the task specification

# Load the data of the ego agent
tr_ego = np.load('data/intersection/x_scene_' + str(scene) + '.npy')

cursors = [16, 20]

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100     # Number of Monte Carlo runs
tr_oppo_s = np.zeros([M, agents['oppo'].n, N + 1])
tr_pedes_s = np.zeros([M, agents['pedes'].n, N + 1])
samples_oppo = agents['oppo'].basis.eta.sample([M, ])
samples_pedes = agents['pedes'].basis.eta.sample([M, ])

for j in range(M):
    agents['oppo'].param = samples_oppo[:, j]
    agents['pedes'].param = samples_pedes[:, j]

    agents['oppo'].predict(0, N)
    agents['pedes'].predict(0, N)

    tr_oppo_s[j, :, :] = agents['oppo'].states
    tr_pedes_s[j, :, :] = agents['pedes'].states

if True:
    # Visualize the result
    visualize(tr_ego, tr_oppo_s, tr_pedes_s, cursor=cursors[1])

if False:
    # Record the video
    record(tr_ego, tr_oppo_s, tr_pedes_s, scene, Ts=Ts, fps=12)