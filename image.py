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

data_id1 = torch.utils.data.Subset(data, (data.identity[:, 0] == 1).nonzero(as_tuple=True)[0])
dataloader_id1 = torch.utils.data.DataLoader(data_id1)

df_sub = []
for ind, samp in enumerate(dataloader_id1):
    df_sub.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_sub = torch.cat(df_sub)
plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))
