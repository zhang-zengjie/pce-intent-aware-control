import numpy as np
from config.overtaking.params import oppo, mode, N, M, o0
from config.overtaking.functions import visualize, record


xx = np.load('results/case_1/x_mode_' + str(mode) + '.npy')

zz_s = np.zeros([oppo.n, N + 1, M])
samples = oppo.basis.eta.sample([M, ])
for j in range(M):
    oppo.param = [samples[0, j], samples[1, j], 1]
    oppo.predict(0, N)
    zz_s[:, :, j] = oppo.states

visualize(xx[:, :N-1], zz_s[:, :N-1, :], mode)

# record(xx[:, :10], zz_s[:, :10, :], mode)