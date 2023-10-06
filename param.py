from def_specs import probability_formula as pf
from def_specs import expecation_formula as ef
from def_specs import variance_formula as vf
from stlpy.STL import LinearPredicate
import numpy as np
from commons import base_length, lanes

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

mu_belief = vf(a3, 0.3) & ef(o, c3, -lanes['middle']) & ef(o, c4, -v_lim)

mu_overtake = LinearPredicate(a=a3, b=lanes['slow'] - delta) & LinearPredicate(a=c3, b=-lanes['slow'] - delta) & ef(a1, c1, b)

phi_safe = mu_safe.always(0, N)
phi_belief = mu_belief.always(0, N)
phi_overtake = mu_overtake.eventually(0, N)

phi = phi_safe & (phi_belief.negation() | phi_overtake)
