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

use_sys = False
if use_sys:
    person_id = int(sys.argv[1])
else:
    person_id = 999

celeb_id = data.identity.unique()[person_id]
data_sub = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id).nonzero(as_tuple=True)[0])
dataloader_sub = torch.utils.data.DataLoader(data_sub)

df_sub = []
for ind, samp in enumerate(dataloader_sub):
    df_sub.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_sub = torch.cat(df_sub)
# plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))
# plt.savefig('image/true_' + str(person_id) + '.png')

d_max = df_sub.shape[0]

d = d_max

n_iter = 300

reg = 1e-5
reg_phi = 1e-5
n_layers = 2
d_hid = 3

obj = []
t_step = 16
dt = 1 / t_step
sd = 1

m = torch.distributions.normal.Normal(0., sd)
bn = transforms.Normalize((0., 0., 0.), (1., 1., 1.))

comp_joint = False

if comp_joint:

    gen = nn_framework.NeuralNetwork2D(3 * d, 3, d_hid, n_layers=n_layers)

    err_ent = nn_framework.NeuralVol2D(3, 3, d_hid, n_layers=n_layers)

    err_kl = nn_framework.NeuralNetwork2D(3 * d, 3 * d, d_hid, n_layers=n_layers)

    param = list(gen.parameters()) + list(err_ent.parameters()) + list(err_kl.parameters())

    opt = torch.optim.Adam(param, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    for n in range(n_iter):
        l = torch.tensor(0.)
        if d == d_max:
            joint_raw = df_sub.flatten(start_dim=0, end_dim=1)
        else:
            joint_raw = df_sub[np.random.choice(np.arange(d_max), replace=True, size=d)].flatten(start_dim=0, end_dim=1)
        joint_raw = joint_raw.unsqueeze(0)
        joint_raw = torch.clamp(joint_raw, min=0.0001, max=0.9999)
        joint = torch.logit(joint_raw)
        for it in range(t_step):
            out_kl = err_kl(joint)
            joint = joint + out_kl * dt
            l = l + reg_phi * out_kl.pow(2).mean() * dt
        z = gen(joint)
        for it in range(t_step):
            out = reg * torch.sigmoid(err_ent(z))
            z = z + out * m.sample(z.shape) * np.sqrt(dt)
            l = l - reg * out.pow(2).mean() * dt
        z = torch.sigmoid(z)
        l = l + (joint_raw - torch.tile(z, dims=[1, d, 1, 1])).pow(2).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
        obj.append(float(l))
        print('obj = {0:0.5f} at iteration {1:n}'.format(float(obj[-1]), n))

else:

    gen = nn_framework.NeuralNetwork2D(3, 3, d_hid, n_layers=n_layers)

    err_ent = nn_framework.NeuralVol2D(3, 3, d_hid, n_layers=n_layers)

    err_kl = nn_framework.NeuralNetwork2D(3 * d, 3 * d, d_hid, n_layers=n_layers)

    param = list(gen.parameters()) + list(err_ent.parameters()) + list(err_kl.parameters())

    opt = torch.optim.Adam(param, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    m_noise = torch.distributions.uniform.Uniform(0, 1)
    noise_dim = torch.tensor(df_sub.shape)
    noise_dim[0] = 1

    for n in range(n_iter):
        l = torch.tensor(0.)
        if d == d_max:
            joint_raw = df_sub.flatten(start_dim=0, end_dim=1)
        else:
            joint_raw = df_sub[np.random.choice(np.arange(d_max), replace=True, size=d)].flatten(start_dim=0, end_dim=1)
        joint_raw = joint_raw.unsqueeze(0)
        joint_raw = torch.clamp(joint_raw, min=0.0001, max=0.9999)
        joint = torch.logit(joint_raw)
        for it in range(t_step):
            out_kl = err_kl(joint)
            joint = joint + out_kl * dt
            l = l + reg_phi * out_kl.pow(2).mean() * dt
        z = torch.logit(m_noise.sample(noise_dim))
        for it in range(t_step):
            drift = gen(z)
            out = reg * torch.sigmoid(err_ent(z))
            z = z + drift * dt + out * m.sample(z.shape) * np.sqrt(dt)
            l = l - reg * out.pow(2).mean() * dt
        z = torch.sigmoid(z)
        l = l + (joint_raw - torch.tile(z, dims=[1, d, 1, 1])).pow(2).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
        obj.append(float(l))
        print('obj = {0:0.5f} at iteration {1:n}'.format(float(obj[-1]), n))

plt.figure()
plt.imshow(z.squeeze(0).detach().numpy().transpose(1,2,0))
# plt.savefig('image/sim_n' + str(n_iter) + '_' + str(person_id) + '.png')

print('-----process takes {:0.6f} seconds-----'.format(time.time() - start))

