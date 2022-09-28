import numpy as np
import torch
from torch import nn

import nn_framework

import ot_num

import time
start = time.time()

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

### optimal transport

# balanced (approx)
mm_ot_eq = ot_num.multimarg_unbalanced_ot(p0, p1, p2, costm=costm, reg=reg, reg_phi=10000, coeff=coeff, n_iter=1000)

# unbalanced
mm_ot_neq = ot_num.multimarg_unbalanced_ot(p0, p1, p2, costm=costm, reg=reg, reg_phi=reg_phi, coeff=coeff, n_iter=1000)

### neural network

softmax = nn.Softmax(dim=1)

# balanced
mod_eq = nn_framework.NeuralNetwork(3 * 4, 4, 100, n_layers=16)
opt_eq = torch.optim.Adam(mod_eq.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
obj_eq = []
for n in range(1000):
    x_sim = torch.zeros((100, 3))
    x_sim[:, 0] = torch.multinomial(torch.tensor(p0), 100, replacement=True)
    x_sim[:, 1] = torch.multinomial(torch.tensor(p1), 100, replacement=True)
    x_sim[:, 2] = torch.multinomial(torch.tensor(p2), 100, replacement=True)
    inp = torch.zeros((100, 3, 4))
    for i in range(100):
        inp[i, torch.arange(3).long(), x_sim[i].long()] = 1
    output = softmax(mod_eq(inp))
    l = torch.einsum('bij,jk->bik', inp.double(), torch.tensor(costm))
    l = torch.einsum('bij,bj->bi', l, output.double())
    l = l.sum(axis=1).mean()
    l = l + output
    opt_eq.zero_grad()
    l.backward()
    opt_eq.step()
    obj_eq.append(float(l))

print('-----process takes {:0.6f} seconds-----'.format(time.time() - start))