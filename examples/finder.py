import random

import torch as th
import numpy as np
import xarray as xr

import sphaera as sph

sph.default_device = 0


import lightning as L
import torch.nn.functional as F
import torch.utils.data as D

from sphaera.core3d.gridsys.regular3 import RegularGrid
from sphaera.core3d.vec3 import dot, norm, cross, normalize, mult

import torch._dynamo

torch._dynamo.config.suppress_errors = True


def cast(data):
    d = np.array(data, dtype=np.float64).reshape(181, 360)
    return sph.cast(np.concatenate((d[:, 359:351:-1], d, d[:, 0:8:1]), axis=1), device=0).reshape(1, 1, 181, 376, 1)


def strip(data):
    d = data.reshape(1, 1, 181, 376, 1)
    return d[:, :, :, 8:368, 0:1]

# --------------------------------
# Step 1: Setup grid system
# --------------------------------

sph.bind(RegularGrid(
    basis='lng,lat,alt',
    W=376, L=181, H=2,
    east=0 - 1 * 8, west=360 + 1 * 8,
    north=-89.999, south=89.999,
    upper=0.99, lower=1.01
))
sph.use('x,y,z')
sph.use('xyz')
sph.use('theta,phi,r')
sph.use('thetaphir')

if sph.mps_ready or sph.cuda_ready:
    sph.set_device(0)
else:
    sph.set_device(-1)


fx = sph.align(sph.xyz.x[0][:, :, :, :, 0:1])
fy = sph.align(sph.xyz.y[1][:, :, :, :, 0:1])
fz = sph.align(sph.xyz.z[2][:, :, :, :, 0:1])
a = cast(2 * np.random.random([181, 360]) - 1)
ux = cast(2 * np.random.random([181, 360]) - 1)
uy = cast(2 * np.random.random([181, 360]) - 1)
vx = cast(2 * np.random.random([181, 360]) - 1)
vy = cast(2 * np.random.random([181, 360]) - 1)

# ----------------------------------------------
# Step 2: Define a machine learning model
# ----------------------------------------------

class BestFinder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.a = th.nn.Parameter(a).to(th.device('mps'))
        self.ux = th.nn.Parameter(ux).to(th.device('mps'))
        self.uy = th.nn.Parameter(ux).to(th.device('mps'))
        self.vx = th.nn.Parameter(vx).to(th.device('mps'))
        self.vy = th.nn.Parameter(vy).to(th.device('mps'))
        self.u = (self.ux * fx, self.uy * fy, fz)
        self.v = (self.vx * fx, self.vy * fy, fz)

    def forward(self, x):
        ix, jx, dd, theta = x
        a0 = self.a[:, :, jx, ix, 0:1]
        u0 = self.ux[:, :, jx, ix, 0:1], self.uy[:, :, jx, ix, 0:1], fz[:, :, jx, ix, 0:1]
        v0 = self.vx[:, :, jx, ix, 0:1], self.vy[:, :, jx, ix, 0:1], fz[:, :, jx, ix, 0:1]
        ds = 2 * th.pi / 360 * dd # dd degree distance

        lat = 90 - jx
        lng = th.fmod(ix - 8, 360)
        lambd = (90 - lat) / 180 * th.pi
        theta = th.atan2(u0[1], u0[0]) + theta
        eta = th.acos(th.cos(ds) * th.cos(lambd) + th.sin(ds) * th.sin(lambd) * th.cos(theta))
        alpha = th.atan2(2 * th.sin(lambd) * th.tan(theta / 2), th.tan(theta / 2) * th.tan(theta / 2) * th.sin(lambd + ds) + th.sin(lambd - ds))
        ix = th.fmod(8 + lng + alpha * 180 / th.pi, 360).long()
        jx = (eta * 180 / th.pi).long()

        return u0, v0, a0 + (th.cos(theta) + a0 * th.sin(theta)) * ds, self.a[:, :, jx, ix, 0:1]

    def training_step(self, batch, batch_idx):
        paths = batch
        u0, v0, a_exp, a_real = self.forward(paths)
        dot_ulen = dot(u0, u0)
        dot_vlen = dot(v0, v0)
        dot_orth = dot(u0, v0)
        ulen_loss = F.mse_loss(dot_ulen, th.ones_like(dot_ulen))
        vlen_loss = F.mse_loss(dot_vlen, th.ones_like(dot_vlen))
        orth_loss = F.mse_loss(dot_orth, th.zeros_like(dot_orth))
        asgn_loss = F.mse_loss(a_real, a_exp)
        loss = ulen_loss + vlen_loss + orth_loss + asgn_loss
        self.log("ulen_loss", ulen_loss, prog_bar=True, logger=True)
        self.log("vlen_loss", vlen_loss, prog_bar=True, logger=True)
        self.log("orth_loss", orth_loss, prog_bar=True, logger=True)
        self.log("asgn_loss", asgn_loss, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ----------------------------------------------
# Step 3: Define a dataset
# ----------------------------------------------

class RandomPathDataset(D.dataset.Dataset):
    def __getitem__(self, index):
        return random.sample(range(376), 1)[0], random.sample(range(181), 1)[0], random.sample(range(2, 10), 1)[0], random.random() * np.pi * 2

    def __len__(self):
        return 376 * 181 * 4


dataset = RandomPathDataset()
train, valid = th.utils.data.random_split(dataset, [376 * 181 * 3, 376 * 181])

# -------------------
# Step 3: Train
# -------------------
finder = BestFinder()
trainer = L.Trainer()
trainer.fit(finder, th.utils.data.DataLoader(train, batch_size=128), th.utils.data.DataLoader(valid, batch_size=1))

th.save(finder.a, 'a.dat')
th.save(finder.ux, 'ux.dat')
th.save(finder.uy, 'uy.dat')
th.save(finder.vx, 'vx.dat')
th.save(finder.vy, 'vy.dat')

