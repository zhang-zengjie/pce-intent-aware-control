import numpy as np
from config.overtaking.params import initialize
from config.overtaking.functions import visualize, record

# First of first, choose the scene
scene = 0    # Select the scene of certain intentions: 
            # 0 for a switching-lane OV 
            # 1 for a slowing-down OV
            # 2 for a speeding-up OV
N = 15      # Control horizon
dir = 'data/overtaking/'

print("---------------------------------------------------------")
print('Initializing...')
print("---------------------------------------------------------")
# Initialize system and specification
agents, _ = initialize(scene, N)
            # agents: the dictionary of agents
                # agents['ego']: ego vehicle (EV)
                # agents['oppo']: opponent vehicle (OV)

# Load the data of the ego agent
print("---------------------------------------------------------")
print('Loading data from' + dir)
print("---------------------------------------------------------")
xe = np.load(dir + 'xe_scene_' + str(scene) + '.npy')

# Perform 100 times Monte Carlo sampling for the opponent agent
M = 100                                         # Number of Monte Carlo runs
so = agents['oppo'].basis.eta.sample([M, ])     # Monte Carlo samples
xo = np.zeros([M, agents['oppo'].n, N + 1])     # Sampled trajectories

for j in range(M):
    
    agents['oppo'].param = [so[0, j], so[1, j], 1]
    agents['oppo'].predict(0, N)
    xo[j] = agents['oppo'].states

if False:
    # Visualize the result
    visualize(agents, xe[:, :N-1], xo[:, :, :N-1], scene)

if True:
    # Record the video
    record(agents, xe[:, :N-1], xo[:, :, :N-1], scene, fps=12)