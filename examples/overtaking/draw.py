import numpy as np
from config import initialize, visualize, record, complexity, data_dir


def main(scene):

    N = 15      # Control horizon
    
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
    print('Loading data from' + data_dir)
    print("---------------------------------------------------------")
    agents['ego'].states = np.load(data_dir + '/xe_scene_' + str(scene) + '.npy')

    return agents


def draw(agents, scene):

    xe = agents['ego'].states
    oppo = agents['oppo']
    # Perform 100 times Monte Carlo sampling for the opponent agent
    M = 100                                         # Number of Monte Carlo runs
    N = oppo.N
    so = oppo.basis.eta.sample([M, ])     # Monte Carlo samples
    xo = np.zeros([M, oppo.n, N + 1])     # Sampled trajectories

    for j in range(M):
        
        oppo.param = [so[0, j], so[1, j], 1]
        oppo.predict(0, N)
        xo[j] = oppo.states

    if True:
        # Visualize the result
        visualize(agents, xe[:, :N-1], xo[:, :, :N-1], scene)

    if False:
        # Record the video
        record(agents, xe[:, :N-1], xo[:, :, :N-1], scene, fps=24)

    if False:
        # Visualize complexity analysis
        complexity(data_dir)


if __name__ == "__main__":

    # First of first, choose the mode
    intent = 2    # Select the intent of OV: 
                # 0 for switching-lane
                # 1 for slowing-down
                # 2 for speeding-up

    agents = main(intent)
    draw(agents, intent)
    