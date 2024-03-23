import numpy as np
from config.intersection.params import initialize
from config.intersection.functions import visualize, record

# First of first, choose the scenario
scene = 1    # Select simulation scenario: 
        # 0 for no_reaction 
        # 1 for reaction with proposed method
N = 35

# Initialize system and specification
sys, phi = initialize(scene, N)
            # sys: the dictionary of agents
            # phi: the task specification

# Load the data of the ego agent
tr_ego = np.load('data/intersection/x_scene_' + str(scene) + '.npy')

cursors = [16, 24]

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100     # Number of Monte Carlo runs
tr_oppo_s = np.zeros([M, sys['oppo'].n, N + 1])
tr_pedes_s = np.zeros([M, sys['pedes'].n, N + 1])
samples_oppo = sys['oppo'].basis.eta.sample([M, ])
samples_pedes = sys['pedes'].basis.eta.sample([M, ])

for j in range(M):
    sys['oppo'].param = samples_oppo[:, j]
    sys['pedes'].param = samples_pedes[:, j]

    sys['oppo'].predict(0, N)
    sys['pedes'].predict(0, N)

    tr_oppo_s[j, :, :] = sys['oppo'].states
    tr_pedes_s[j, :, :] = sys['pedes'].states

if True:
    # Visualize the result
    visualize(tr_ego, tr_oppo_s, tr_pedes_s, cursor=cursors[0])

if False:
    # Record the video
    record(tr_ego, tr_oppo_s, tr_pedes_s, scene, Ts=Ts, fps=12)