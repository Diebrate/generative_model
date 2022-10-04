import numpy as np
import matplotlib.pyplot as plt

import os

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import nn_framework

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

celeb_id = data.identity.unique()[666]
data_sub = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id).nonzero(as_tuple=True)[0])
dataloader_sub = torch.utils.data.DataLoader(data_sub)

df_sub = []
for ind, samp in enumerate(dataloader_sub):
    df_sub.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_sub = torch.cat(df_sub)
plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))

d = df_sub.shape[0]

n_iter = 500

reg = 0.01
reg_phi = 10

gen = nn_framework.NeuralNetwork2D(3, 3, 100, n_layers=4)
opt_gen = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_ent = nn_framework.NeuralVol2D(3, 3, 100, n_layers=4)
opt_ent = torch.optim.Adam(err_ent.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

err_kl = nn_framework.NeuralVol2D(3, 3, 100, n_layers=4)
opt_kl = torch.optim.Adam(err_kl.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

m = torch.distributions.normal.Normal(0., 1.)

obj = []

for n in range(n_iter):

