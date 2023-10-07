import numpy as np
from itertools import product
import chaospy as cp
import numpoly


def get_mean_from_pce(zeta_hat):
    N = len(zeta_hat)
    E = np.zeros([N, 4])
    for i in range(N):
        E[i] = zeta_hat[i][0]
    return E


def get_var_from_pce(zeta_hat, basis, eta):
    N = zeta_hat.shape[0]
    L = zeta_hat.shape[1]
    S = zeta_hat.shape[2]
    Var = np.zeros([N, 4])
    for i, j in product(range(N), range(S)):
        Var[i][j] = sum([zeta_hat[i][k][j] ** 2 * cp.E(basis[k] ** 2, eta) for k in range(1, L)])
    return Var
