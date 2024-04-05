import numpy as np
from config import initialize, visualize, record, complexity, data_dir


def main(scene):

    N = 35
    
    print("---------------------------------------------------------")
    print('Initializing...')
    print("---------------------------------------------------------")
    # Initialize system and specification
    agents, _ = initialize(scene, N)
                # agents: the dictionary of agents

    # Load the data of the ego agent
    print("---------------------------------------------------------")
    print('Loading data from ' + data_dir)
    print("---------------------------------------------------------")
    agents['ego'].states = np.load(data_dir + '/xe_scene_' + str(scene) + '.npy')

    return agents


def draw(agents, scene, step):

    # Perform 100 times Monte Carlo sampling for the opponent agent
    M = 100     # Number of Monte Carlo runs
    N = agents['ego'].N

    xe = agents['ego'].states
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
        visualize(agents, xe, xo, xp, scene, step)

    if False:
        # Record the video
        record(agents, xe, xo, xp, scene, fps=24)

    if False:
        # Visualize complexity analysis
        complexity(data_dir)


if __name__ == "__main__":

    # First of first, choose the scenario
    scene = 0       # 0 for no awareness
                    # 1 for intention-aware
    # Choose the instant of the view
    step = 20       # 16: the step to show relation with the OV
                    # 20: the step to show relation with the pedestrian
    agents = main(scene)

    draw(agents, scene, step)
    