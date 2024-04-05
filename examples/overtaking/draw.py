import numpy as np
from config import visualize


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

    visualize(agents, xe[:, :N-1], xo[:, :, :N-1], scene)
