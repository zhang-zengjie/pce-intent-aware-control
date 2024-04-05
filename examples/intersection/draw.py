import numpy as np
from config import initialize, visualize, record, complexity
import os

def main(scene):

    N = 35
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    print("---------------------------------------------------------")
    print('Initializing...')
    print("---------------------------------------------------------")
    # Initialize system and specification
    agents, _ = initialize(scene, N)
                # agents: the dictionary of agents

    # Load the data of the ego agent
    print("---------------------------------------------------------")
    print('Loading data from ' + dir)
    print("---------------------------------------------------------")
    xe = np.load(dir + '/xe_scene_' + str(scene) + '.npy')

    cursors = [16, 20]

    draw(xe, agents, cursors)


def draw(xe, agents, cursors):

    # Perform 100 times Monte Carlo sampling for the opponent agent
    M = 100     # Number of Monte Carlo runs
    N = agents['ego'].N
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

    if True:
        # Visualize the result
        visualize(agents, xe, xo, xp, cursor=cursors[1])

    if False:
        # Record the video
        record(agents, xe, xo, xp, scene, fps=24)

    if False:
        # Visualize complexity analysis
        complexity(dir)


if __name__ == "__main__":

    # First of first, choose the mode
    intent = 1    # Select the intent of OV: 
                # 0 for switching-lane
                # 1 for slowing-down
                # 2 for speeding-up
    main(intent)