import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import nn_framework

import time
start = time.time()

data = 'CelebA'
path = 'data/' + data
load = getattr(torchvision.datasets, data)

# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0., 0., 0.), (1., 1., 1.))])

if not os.path.exists(path):
    os.mkdir(path)

if len(os.listdir(path)) > 0:
    data = load(path, download=False, transform=transform)
else:
    data = load(path, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(data)

use_sys = True
if use_sys:
    person_id = int(sys.argv[1])
else:
    person_id = 123

celeb_id = data.identity.unique()[person_id]
data_sub = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id).nonzero(as_tuple=True)[0])
dataloader_sub = torch.utils.data.DataLoader(data_sub)

df_sub = []
for ind, samp in enumerate(dataloader_sub):
    df_sub.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_sub = torch.cat(df_sub)
plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))
plt.savefig('image/true_' + str(person_id) + '.png')

d = df_sub.shape[0]

n_iter = 5000

reg = 0.001
reg_phi = 0.001
n_layers = 4
d_hid = 8

gen = nn_framework.NeuralNetwork2D(3 * d, 3, d_hid, n_layers=n_layers)
# opt_gen = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_ent = nn_framework.NeuralVol2D(3, 3, d_hid, n_layers=n_layers)
# opt_ent = torch.optim.Adam(err_ent.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_kl = nn_framework.NeuralVol2D(3 * d, 3 * d, d_hid, n_layers=n_layers)
# opt_kl = torch.optim.Adam(err_kl.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

opt = torch.optim.Adam(list(gen.parameters()) + list(err_ent.parameters()) + list(err_kl.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

m = torch.distributions.normal.Normal(0., 1.)
bn = transforms.Normalize((0., 0., 0.), (1., 1., 1.))

obj = []
t_step = 10
dt = 1 / t_step

for n in range(n_iter):
    l = torch.tensor(0.)
    joint = df_sub.flatten(start_dim=0, end_dim=1)
    joint = joint.unsqueeze(0)
    for it in range(t_step):
        out_kl = nn.functional.relu(err_kl(joint))
        joint = joint + out_kl * m.sample(joint.shape) * np.sqrt(dt)
        l = l + reg_phi * out_kl.pow(2).mean() * dt
    z = gen(joint)
    for it in range(t_step):
        out = reg * err_ent(z).abs() / (1 + err_ent(z).abs())
        z = z + out * m.sample(z.shape) * np.sqrt(dt)
        l = l - reg * out.pow(2).mean() * dt
    l = l + (joint - torch.tile(z, dims=[1, d, 1, 1])).pow(2).mean()
    # opt_kl.zero_grad()
    # opt_gen.zero_grad()
    # opt_ent.zero_grad()
    opt.zero_grad()
    l.backward()
    # opt_kl.step()
    # opt_gen.step()
    # opt_ent.step()
    opt.step()
    obj.append(float(l))
    print('obj = {0:0.5f} at iteration {1:n}'.format(float(obj[-1]), n))

plt.figure()
plt.imshow(bn(z).squeeze(0).detach().numpy().transpose(1,2,0))
plt.savefig('image/sim_n' + str(n_iter) + '_' + str(person_id) + '.png')

print('-----process takes {:0.6f} seconds-----'.format(time.time() - start))

