import numpy as np

import nn_framework

import ot_num

np.random.seed(111)

coeff = np.ones(3) / 3

p0 = np.array([0.1, 0.2, 0.3, 0.4])
p1 = np.array([0.25, 0.25, 0.25, 0.25])
p2 = np.array([0.4, 0.4, 0.1, 0.1])

costm = np.random.uniform(size=(4, 4))
costm = costm + costm.T
np.fill_diagonal(costm, 0)

reg = 100
reg_phi = 1

# balanced (approx)
mm_ot_eq = ot_num.multimarg_unbalanced_ot(p0, p1, p2, costm=costm, reg=reg, reg_phi=10000, coeff=coeff, n_iter=1000)

# unbalanced
mm_ot_neq = ot_num.multimarg_unbalanced_ot(p0, p1, p2, costm=costm, reg=reg, reg_phi=reg_phi, coeff=coeff, n_iter=1000)