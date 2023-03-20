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

person_id2 = 666

if use_gpu:
    cuda = torch.device('cuda')

celeb_id = data.identity.unique()[person_id]
data_sub = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id).nonzero(as_tuple=True)[0])
dataloader_sub = torch.utils.data.DataLoader(data_sub)

celeb_id2 = data.identity.unique()[person_id2]
data_sub2 = torch.utils.data.Subset(data, (data.identity[:, 0] == celeb_id2).nonzero(as_tuple=True)[0])
dataloader_sub2 = torch.utils.data.DataLoader(data_sub2)

df_all = []
for ind, samp in enumerate(dataloader_sub):
    df_all.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_all = torch.cat(df_all)

df_all2 = []
for ind, samp in enumerate(dataloader_sub2):
    df_all2.append(samp[0])
    # plt.imshow(samp[0][0].numpy().transpose((1, 2, 0)))

df_all2 = torch.cat(df_all2)

if use_gpu:
    df_all = df_all.cuda()
    df_all2 = df_all2.cuda()

# plt.imshow(torchvision.utils.make_grid(df_all).numpy().transpose(1, 2, 0))
# plt.savefig('image/true_' + str(person_id) + '.png')

# clear data

df_all = torch.concat((df_all, df_all2), dim=0)
df_all = df_all[np.random.choice(np.arange(df_all.shape[0]), replace=False, size=20)]

d_max = df_all.shape[0]

d = d_max

n_train = int(d_max * 0.7)
n_test = d_max - n_train

ind_mix = np.random.choice(np.arange(d_max), size=d_max, replace=False)
df_sub = df_all[ind_mix[:n_train]]
df_test = df_all[ind_mix[n_train:]]

n_iter = 500

reg = 1e-4
reg_phi = 1e-4
n_layers = 4
d_hid = 512
d_feat = 512
height, width = df_sub[0].shape[1:]
n_noise = 30 - n_train

n_total = n_train + n_noise

obj = []

del dataloader_sub, data_sub, celeb_id, dataloader_sub2, data_sub2, celeb_id2, dataloader, data
gc.collect()

# nn_feat = nn_framework.NeuralFeat(d_in=3, d_out=d_feat, d_hid=d_hid, n_val_layers=n_layers, n_same_layers=n_layers)
# nn_class = nn.Linear(d_feat, 1, bias=False)

# nn_feat = nn_framework.NNconv5_3(d_in=3, d_out=d_feat, d_hid=d_hid)
nn_feat = nn_framework.NNvgg(d_in=3, d_out=d_feat)
nn_class = nn.Conv2d(d_feat, 1, kernel_size=(1, 1), bias=False)

### nn for coord model
# nn_coord = nn.Conv2d(d_feat, 2, kernel_size=(3, 3), padding='same')
nn_coord = nn_framework.NNconv5_3(d_in=d_feat, d_out=2, d_hid=d_hid)
nn_gen = nn_framework.NNconv5_3(d_in=2, d_out=3, d_hid=d_hid, size=1)

if use_gpu:
    nn_feat = nn_feat.cuda()
    nn_class = nn_class.cuda()
    nn_coord = nn_coord.cuda()
    nn_gen = nn_gen.cuda()

param = list(nn_feat.parameters()) + list(nn_class.parameters()) + list(nn_coord.parameters()) + list(nn_gen.parameters())

opt = torch.optim.Adam(param, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

# inp = torch.clamp(df_sub, min=0.0001, max=0.9999)
# inp = torch.logit(inp)

inp = df_sub.clone()

# noise = torch.randn(n_noise, 3, height, width).sigmoid()
noise = df_all2[:n_noise].clone()

if use_gpu:
    inp = inp.cuda()
    noise = noise.cuda()
    df_test = df_test.cuda()

inp = torch.concat((inp, noise), dim=0)
target = torch.concat((torch.ones(df_sub.shape[0]), torch.zeros(n_noise)), dim=0)

if use_gpu:
    target = target.cuda()

ind = np.random.permutation(np.arange(n_total))

inp = inp[ind]
target = target[ind]

ind_img = np.arange(n_total)[target.cpu() == 1]
ind_noise = np.arange(n_total)[target.cpu() == 0]

loss = nn.BCELoss()

# pool = nn.AdaptiveAvgPool2d((1, 1), divisor_override=1)

h, w = nn_feat(inp).detach().shape[2:]
# pool = nn.AvgPool2d((h, w))

gc.collect()

n = 0
while n < n_iter:

    l = torch.tensor(0.).cuda()

    feat = nn_feat(inp) # size n * d_feat * h * w
    # feat_pool = pool(feat).flatten(start_dim=1)
    out_class = nn_class(feat) # size n * 1 * h * w
    out = out_class.mean(dim=(1, 2, 3)).sigmoid()

    feat_avg = feat.mean(dim=(2, 3))

    l = l + loss(out, target)

    # var model

    feat_pool = feat_avg[ind_img]

    l = l + feat_pool.T.cov().det()

    # feat_pool2 = feat_avg[ind_noise]

    # l = l + feat_pool2.T.cov().det()

    # coord model

    coord = nn_coord(feat).sigmoid() # size n * 2 * h * w
    img_c = nn_gen(coord).sigmoid() # size n * 3 * h * w

    l = l + ((img_c - inp).pow(2).mean(dim=1) * (out_class.squeeze().sigmoid())).mean()

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
    plt.imshow(torchvision.utils.make_grid(df_all).cpu().numpy().transpose(1, 2, 0))
else:
    plt.imshow(torchvision.utils.make_grid(df_all).numpy().transpose(1, 2, 0))
plt.tight_layout()
plt.savefig('image/class/true.png')

# plt.figure()
# plt.imshow(torchvision.utils.make_grid(torch.sigmoid(out)).numpy().transpose(1, 2, 0))

fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(16, 16))

up = nn.Upsample(size=(height, width))

res_all = up(out_class.detach())
for i in range(n_total):
    # res = nn_class(feat[i]).squeeze().detach()
    res = res_all[i].squeeze()
    res = (res - res.min()) / (res.max() - res.min())
    if use_gpu:
        res = res.cpu()
    axs[i // 10, i % 10].imshow(res.numpy())
    axs[i // 10, i % 10].axis('off')
plt.tight_layout()
plt.savefig('image/class/train.png')

fig_test, axs_test = plt.subplots(ncols=n_test, figsize=(16, 8))

feat_test = nn_feat(df_test) # size n * d_feat * h * w
out_class_test = nn_class(feat_test) # size n * 1 * h * w
res_test_all = up(out_class_test.detach())
for i in range(n_test):
    res_test = res_test_all[i].squeeze()
    res_test = (res_test - res_test.min()) / (res_test.max() - res_test.min())
    if use_gpu:
        res_test = res_test.cpu()
    axs_test[i].imshow(res_test.numpy())
    axs_test[i].axis('off')
plt.tight_layout()
plt.savefig('image/class/test.png')

ref = np.random.randn(1, 2, height, width)
for i in range(height):
    for j in range(width):
        ref[0][0][i, j] = i / height
        ref[0][1][i, j] = j / width
ref = torch.tensor(ref)
if use_gpu:
    ref = ref.cuda()

plt.figure()
center = nn_gen(ref)
if use_gpu:
    center = center.cpu()
plt.imshow(center.detach().squeeze().numpy().transpose(1, 2, 0))
plt.tight_layout()
plt.savefig('image/class/center.png')

print(out)
print(out_class_test.mean(dim=(1, 2, 3)).sigmoid())
print(target)
print(float(loss(out, target)))