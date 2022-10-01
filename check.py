import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

import nn_framework

import ot_num

import time
start = time.time()

np.random.seed(111)

reg = 0.001
reg_phi = 1

N = 1000
n_sim = 100

x1 = np.random.multivariate_normal(mean=[-5, 0], cov=[[1, 0], [0, 1]], size=N)
x2 = np.random.multivariate_normal(mean=[5, 0], cov=[[1, 0], [0, 1]], size=N)

gen = nn_framework.NeuralNetwork(4, 2, 100, n_layers=4)
opt_gen = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_ent = nn_framework.NeuralNetwork(3, 1, 100, n_layers=4)
opt_ent = torch.optim.Adam(err_ent.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_kl = nn_framework.NeuralNetwork(5, 2, 100, n_layers=4)
opt_kl = torch.optim.Adam(err_kl.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                               torch.tensor(np.diag([1, 1])).float())
obj = []

t_step = 20
dt = 1 / t_step
t_list = np.linspace(0, 1, t_step + 1)

for n in range(500):
    l = torch.tensor(0.)
    x1_temp = torch.tensor(x1[np.random.choice(np.arange(N), replace=True, size=n_sim)])
    x2_temp = torch.tensor(x2[np.random.choice(np.arange(N), replace=True, size=n_sim)])
    for it in range(t_step):
        inp_kl = torch.cat([x1_temp, x2_temp, t_list[it] * torch.ones(n_sim, 1)], dim=1)
        out_kl = nn.functional.relu(err_kl(inp_kl))
        out1 = out_kl[:, 0].reshape(-1, 1)
        out2 = out_kl[:, 1].reshape(-1, 1)
        x1_temp = x1_temp + out1 * m.sample([n_sim]) * np.sqrt(dt)
        x2_temp = x2_temp + out2 * m.sample([n_sim]) * np.sqrt(dt)
        l = l + 0.5 * reg_phi * out1.pow(2).mean() * dt + 0.5 * reg_phi * out2.pow(2).mean() * dt
    x = torch.cat([x1_temp, x2_temp], dim=1)
    y = gen(x)
    for it in range(t_step):
        inp = torch.cat([y, t_list[it] * torch.ones(n_sim, 1)], dim=1)
        out = reg * err_ent(inp).abs() / (1 + err_ent(inp).abs())
        y = y + out * m.sample([n_sim]) * np.sqrt(dt)
        l = l - reg * out.pow(2).mean() * dt
    l = l + (x1_temp - y).pow(2).sum(axis=1).mean()
    l = l + (x2_temp - y).pow(2).sum(axis=1).mean()
    opt_kl.zero_grad()
    opt_gen.zero_grad()
    opt_ent.zero_grad()
    l.backward()
    opt_kl.step()
    opt_gen.step()
    opt_ent.step()
    obj.append(float(l))
    print('obj = {0:0.5f} at iteration {1:n}'.format(float(obj[-1]), n))

df = np.vstack((x1[np.random.choice(np.arange(N), replace=True, size=n_sim)],
                x2[np.random.choice(np.arange(N), replace=True, size=n_sim)],
                y.detach().numpy()))
df = pd.DataFrame(df, columns=['x', 'y'])
df['group'] = np.repeat(['s1', 's2', 'b'], n_sim)
sns.scatterplot(x='x', y='y', data=df, hue='group', linewidth=0, s=3)

print('-----process takes {:0.6f} seconds-----'.format(time.time() - start))