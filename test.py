import numpy as np
import matplotlib.pyplot as plt

import os

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import nn_framework

data = 'MNIST'
path = 'data/' + data
load = getattr(torchvision.datasets, data)

transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0., 0., 0.), (1., 1., 1.))])

if not os.path.exists(path):
    os.mkdir(path)

if len(os.listdir(path)) > 0:
    data = load(path, download=False, train=True, transform=transform)
else:
    data = load(path, download=True, train=True, transform=transform)

# dataloader = torch.utils.data.DataLoader(data)

x = torch.stack([i[0] for i in data])
labels = torch.tensor([i[1] for i in data])
label_list = data.classes

pick = 6
z = x[labels == 6]

gen = nn_framework.NeuralNetwork2D(1, 1, 10, n_layers=50)