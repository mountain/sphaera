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
from sphaera.core3d.vec3 import dot

import torch._dynamo

torch._dynamo.config.suppress_errors = True

if sph.mps_ready or sph.cuda_ready:
    sph.set_device(0)
else:
    sph.set_device(-1)


def cast(data):
    d = np.array(data, dtype=np.float32).reshape(181, 360)
    return sph.cast(np.concatenate((d[:, 359:351:-1], d, d[:, 0:8:1]), axis=1), device=0).reshape(1, 1, 181, 376)


def strip(data):
    d = data.reshape(1, 1, 181, 376)
    return d[:, :, :, 8:368]


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


a = cast(2 * np.random.random([181, 360]) - 1)
ux = cast(2 * np.random.random([181, 360]) - 1)
uy = cast(2 * np.random.random([181, 360]) - 1)
uz = cast(np.zeros([181, 360]))
vx = cast(2 * np.random.random([181, 360]) - 1)
vy = cast(2 * np.random.random([181, 360]) - 1)
vz = cast(np.zeros([181, 360]))

# ----------------------------------------------
# Step 2: Define a machine learning model
# ----------------------------------------------

class BestFinder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.a = th.nn.Parameter(a)
        self.ux = th.nn.Parameter(ux)
        self.uy = th.nn.Parameter(ux)
        self.uz = th.nn.Parameter(uz)
        self.vx = th.nn.Parameter(vx)
        self.vy = th.nn.Parameter(vy)
        self.vz = th.nn.Parameter(vz)
        self.u = self.ux, self.uy, self.uz
        self.v = self.vx, self.vy, self.vz

    def forward(self, x):
        ix, jx, dd, theta = x
        theta = theta.float()
        ds = 2 * th.pi / 360 * dd # dd degree distance
        aexp = self.a + (th.cos(theta) + self.a * th.sin(theta)) * ds

        lat = th.reshape(90 - jx, [1, 1, 181, 1])
        lng = th.reshape(th.fmod(ix - 8, 360), [1, 1, 1, 376])
        lambd = (90 - lat) / 180 * th.pi
        theta = th.atan2(self.u[1], self.u[0]) + theta
        eta = th.acos(th.cos(ds) * th.cos(lambd) + th.sin(ds) * th.sin(lambd) * th.cos(theta))
        alpha = th.atan2(2 * th.sin(lambd) * th.tan(theta / 2), th.tan(theta / 2) * th.tan(theta / 2) * th.sin(lambd + ds) + th.sin(lambd - ds))

        ix = th.fmod(8 + lng + alpha * 180 / th.pi, 360).long()[0, 0]
        jx = (eta * 180 / th.pi).long()[0, 0]
        areal = self.a[:, :, jx, ix]

        return self.u, self.v, aexp, areal

    def training_step(self, batch, batch_idx):
        paths = batch
        u, v, a_exp, a_real = self.forward(paths)
        dot_ulen = dot(u, u)
        dot_vlen = dot(v, v)
        dot_orth = dot(u, v)
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
        xx, yy = np.arange(376), np.arange(181)
        return xx, yy, np.random.random([181, 376]) * 10, np.random.random([181, 376]) * np.pi * 2

    def __len__(self):
        return 376 * 181 * 4


dataset = RandomPathDataset()
train, valid = th.utils.data.random_split(dataset, [376 * 181 * 3, 376 * 181])

# -------------------
# Step 3: Train
# -------------------
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    finder = BestFinder()
    trainer = L.Trainer()
    train_loader = th.utils.data.DataLoader(train, batch_size=1, num_workers=1)
    valid_loader = th.utils.data.DataLoader(valid, batch_size=1, num_workers=1)

    trainer.fit(finder, train_loader, valid_loader)

    th.save(finder.a, 'a.dat')
    th.save(finder.ux, 'ux.dat')
    th.save(finder.uy, 'uy.dat')
    th.save(finder.vx, 'vx.dat')
    th.save(finder.vy, 'vy.dat')

