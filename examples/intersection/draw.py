import numpy as np
from config import visualize


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

    visualize(agents, xe, xo, xp, cursor=cursors[1])
