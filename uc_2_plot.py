import numpy as np
from config.intersection.params import sys, mode, N, M
from config.intersection.functions import visualize, record

tr_ego = np.load('results/case_2/x_mode_' + str(mode) + '.npy')

cursors = [16, 24]

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

visualize(tr_ego, tr_oppo_s, tr_pedes_s, cursor=cursors[0])

# record(tr_ego, tr_oppo_s, tr_pedes_s, mode, Ts=Ts, fps=12)