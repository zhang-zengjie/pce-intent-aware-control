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
xe = np.load('data/intersection/xe_scene_' + str(scene) + '.npy')

cursors = [16, 20]

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100     # Number of Monte Carlo runs
xo = np.zeros([M, agents['oppo'].n, N + 1])
xp = np.zeros([M, agents['pedes'].n, N + 1])
so = agents['oppo'].basis.eta.sample([M, ])
sp = agents['pedes'].basis.eta.sample([M, ])

for j in range(M):
    
    agents['oppo'].param = so[:, j]
    agents['pedes'].param = sp[:, j]

    agents['oppo'].predict(0, N)
    agents['pedes'].predict(0, N)

    xo[j] = agents['oppo'].states
    xp[j] = agents['pedes'].states

if False:
    # Visualize the result
    visualize(agents, xe, xo, xp, cursor=cursors[1])

if True:
    # Record the video
    record(agents, xe, xo, xp, scene, fps=12)