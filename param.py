from pce_specs import probability_formula as pf
from pce_specs import expecation_formula as ef
from pce_specs import variance_formula as vf
from pce_specs import neg_variance_formula as nvf
from stlpy.STL import LinearPredicate
import numpy as np
import numpoly
from gen_basis import base_length, base_sampling_time


N = 20

lanes = {'right': 0,
         'slow': 1.2,
         'middle': 2.4,
         'fast': 1.8,
         'left': 4.8}

a_hat = np.load('a_hat.npy')
psi = np.load('psi.npy')
basis = numpoly.load('basis.npy')
eps = 0.05
delta = 0.5
b = base_length
v_lim = 1.67
o = np.zeros((4, ))

a1 = np.array([1, 0, 0, 0])
c1 = np.array([-1, 0, 0, 0])

a2 = np.array([-1, 0, 0, 0])
c2 = np.array([1, 0, 0, 0])

a3 = np.array([0, 1, 0, 0])
c3 = np.array([0, -1, 0, 0])

a4 = np.array([0, 0, 0, 1])
c4 = np.array([0, 0, 0, -1])

mu_safe = pf(a1, c1, b, eps) | pf(a2, c2, b, eps) | pf(a3, c3, b, eps)

neg_mu_belief = nvf(a3, 0.3) | ef(o, a3, lanes['middle']) | ef(o, a4, v_lim)

mu_overtake = ef(a3, o, lanes['slow'] - delta) & ef(c3, o, -lanes['slow'] - delta) & ef(a1, c1, b)

phi_safe = mu_safe.always(0, N)
phi_belief = neg_mu_belief.eventually(0, N)
phi_overtake = mu_overtake.eventually(0, N)

phi = phi_safe & (phi_belief | phi_overtake)
