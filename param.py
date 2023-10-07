from pce_specs import probability_formula as pf
from pce_specs import expecation_formula as ef
from pce_specs import variance_formula as vf
from pce_specs import neg_variance_formula as nvf
from stlpy.STL import LinearPredicate
import numpy as np
import numpoly
from gen_basis import base_length, base_sampling_time


N = 30

# gamma = np.linspace(0.01, 0, N-1)
gamma = np.linspace(0, 0, N)
a = np.linspace(0, 0, N)
v = np.array([gamma, a])

lanes = {'right': 0,
         'slow': 0.3,
         'middle': 0.6,
         'fast': 0.9,
         'left': 1.2}

a_hat = np.load('a_hat.npy')
psi = np.load('psi.npy')
basis = numpoly.load('basis.npy')
eps = 0.05
b = base_length
v_lim = 3
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
c1 = np.array([-1, 0, 0, 0])

a2 = np.array([-1, 0, 0, 0])
c2 = np.array([1, 0, 0, 0])

a3 = np.array([0, 1, 0, 0])
c3 = np.array([0, -1, 0, 0])

a4 = np.array([0, 0, 0, 1])
c4 = np.array([0, 0, 0, -1])

a5 = np.array([0, 0, 1, 0])
c5 = np.array([0, 0, -1, 0])

mu_safe = pf(a1, c1, b, eps) | pf(a2, c2, b, eps) | pf(a3, c3, b, eps)

neg_mu_belief = nvf(a3, 13) | ef(o, a3, lanes['middle']) | ef(o, a4, v_lim)

mu_overtake = ef(a3, o, lanes['slow'] - 0.01) & ef(c3, o, - lanes['slow'] - 0.01) & ef(a1, c1, b) & ef(a5, o, - 1e-4) & ef(c5, o, - 1e-4)

phi_safe = mu_safe.always(0, N)
phi_belief = neg_mu_belief.eventually(0, N)
phi_overtake = mu_overtake.eventually(0, N)

# phi = phi_safe & (phi_belief | phi_overtake)

phi = phi_safe & phi_overtake
