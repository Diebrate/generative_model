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

torch.cuda.empty_cache()

import gc
gc.collect()

torch.manual_seed(12345)
np.random.seed(12345)

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
    use_gpu = int(sys.argv[1]) == 0
else:
    person_id = 999
    use_gpu = True

if use_gpu:
    cuda = torch.device('cuda')

celeb_id = data.identity.unique()[person_id]
data_sub = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id).nonzero(as_tuple=True)[0])
dataloader_sub = torch.utils.data.DataLoader(data_sub)

df_sub = []
for ind, samp in enumerate(dataloader_sub):
    df_sub.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_sub = torch.cat(df_sub)

if use_gpu:
    df_sub = df_sub.cuda()

# plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))
# plt.savefig('image/true_' + str(person_id) + '.png')

d_max = df_sub.shape[0]

d = d_max

n_iter = 150

reg = 1e-4
reg_phi = 1e-4
n_layers = 16
d_hid = 64
d_feat = 32
height, width = df_sub[0].shape[1:]
n_noise = 50 - d

n_total = d + n_noise

obj = []

# nn_feat = nn_framework.NeuralNetwork2D(d_in=3, d_out=d_feat, d_hid=d_hid, n_layers=n_layers)
nn_feat = nn_framework.NeuralFeat(d_in=3, d_out=d_feat, d_hid=d_hid, n_val_layers=n_layers, n_same_layers=n_layers)
nn_lin = nn_framework.NeuralLinear(d_in=d_feat, d_out=1)

if use_gpu:
    nn_feat = nn_feat.cuda()
    nn_lin = nn_lin.cuda()

param = list(nn_feat.parameters()) + list(nn_lin.parameters())

opt = torch.optim.Adam(param, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

# inp = torch.clamp(df_sub, min=0.0001, max=0.9999)
# inp = torch.logit(inp)

inp = df_sub.clone()

noise = torch.randn(n_noise, 3, height, width).sigmoid()

if use_gpu:
    inp = inp.cuda()
    noise = noise.cuda()

inp = torch.concat((inp, noise), dim=0)
target = torch.concat((torch.ones(df_sub.shape[0]), torch.zeros(n_noise)), dim=0)

if use_gpu:
    target = target.cuda()

ind = np.random.permutation(np.arange(n_total))

inp = inp[ind]
target = target[ind]

loss = nn.BCELoss()

# pool = nn.AdaptiveAvgPool2d((1, 1), divisor_override=1)

h, w = nn_feat(inp).detach().shape[2:]
pool = nn.AvgPool2d((h, w))

n = 0
while n < n_iter:

    feat = nn_feat(inp)
    feat_pool = pool(feat).flatten(start_dim=1)
    out = torch.sigmoid(nn_lin(feat_pool)).flatten()

    l = loss(out, target)

    opt.zero_grad()
    l.backward()
    opt.step()

    obj.append(float(l))
    print('obj = {0:0.5f} at iteration {1:n}'.format(float(obj[-1]), n))

    if obj[-1] < -1:
        n = n_iter
    else:
        n += 1

plt.figure(figsize=(16, 16))
if use_gpu:
    plt.imshow(torchvision.utils.make_grid(df_sub).cpu().numpy().transpose(1, 2, 0))
else:
    plt.imshow(torchvision.utils.make_grid(df_sub).numpy().transpose(1, 2, 0))
plt.tight_layout()
plt.savefig('image/class/true.png')

# plt.figure()
# plt.imshow(torchvision.utils.make_grid(torch.sigmoid(out)).numpy().transpose(1, 2, 0))

fig, axs = plt.subplots(nrows=5, ncols=10, figsize=(16, 16))

up = nn.Upsample(size=(height, width))

for i in range(n_total):
    h, w = feat[i].shape[1:]
    res = feat[i].flatten(start_dim=1).transpose(0, 1)
    res = nn_lin(res).reshape((h, w)).detach()
    res = res.unsqueeze(0).unsqueeze(0)
    res = up(res).squeeze()
    if use_gpu:
        res = res.cpu()
    axs[i // 10, i % 10].imshow(res.numpy())
    axs[i // 10, i % 10].axis('off')
plt.tight_layout()
plt.savefig('image/class/test.png')