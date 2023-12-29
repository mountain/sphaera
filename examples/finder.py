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
    d = np.array(data, dtype=np.float64).reshape(721, 1440)
    return sph.cast(np.concatenate((d[:, 1439:1407:-1], d, d[:, 0:32:1]), axis=1), device=0).reshape(1, 1, 721, 1504)


def strip(data):
    d = data.reshape(1, 1, 721, 1504)
    return d[:, :, :, 32:1472]


# --------------------------------
# Step 1: Setup grid system
# --------------------------------

sph.bind(RegularGrid(
    basis='lng,lat,alt',
    W=1504, L=721, H=2,
    east=0 - 0.25 * 32, west=360 + 0.25 * 32,
    north=-89.999, south=89.999,
    upper=0.99, lower=1.01
))
sph.use('x,y,z')
sph.use('xyz')
sph.use('theta,phi,r')
sph.use('thetaphir')


a = cast(np.ones([721, 1440]))
ux = cast(np.ones([721, 1440]))
uy = cast(np.zeros([721, 1440]))
uz = cast(np.zeros([721, 1440]))
vx = cast(np.zeros([721, 1440]))
vy = cast(np.ones([721, 1440]))
vz = cast(np.zeros([721, 1440]))

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
        ix, jx = th.meshgrid(ix.flatten(), jx.flatten())
        ix = th.reshape(ix, [1, 1, 721, 1504])
        jx = th.reshape(jx, [1, 1, 721, 1504])
        dd = th.reshape(dd, [1, 1, 721, 1504])
        theta = th.reshape(theta.float(), [1, 1, 721, 1504])
        ds = (th.pi / 180 * dd).float() # dd degree distance
        aexp = self.a + (th.cos(theta) - self.a * th.sin(theta)) * ds

        lat = 90 - jx
        lng = th.fmod(ix - 8, 360)
        lambd = ((90 - lat) / 180 * th.pi).float()
        theta = th.atan2(self.u[1], self.u[0]) + theta
        eta = th.acos(th.cos(ds) * th.cos(lambd) + th.sin(ds) * th.sin(lambd) * th.cos(theta))
        alpha = th.atan2(2 * th.sin(lambd) * th.tan(theta / 2), th.tan(theta / 2) * th.tan(theta / 2) * th.sin(lambd + ds) + th.sin(lambd - ds))

        xx = th.fmod(8 + lng + alpha * 180 / th.pi, 360)[0, 0]
        yy = (eta * 180 / th.pi)[0, 0]
        a0 = self.a[:, :, th.floor(yy).long(), th.floor(xx).long()]
        a1 = self.a[:, :, th.floor(yy).long(), th.ceil(xx).long()]
        a2 = self.a[:, :, th.ceil(yy).long(), th.floor(xx).long()]
        a3 = self.a[:, :, th.ceil(yy).long(), th.ceil(xx).long()]
        a4 = th.lerp(a0, a1, th.ceil(xx) - th.floor(xx))
        a5 = th.lerp(a2, a3, th.ceil(xx) - th.floor(xx))
        a6 = th.lerp(a4, a5, th.ceil(yy) - th.floor(yy))

        return self.u, self.v, self.a, aexp, a6

    def training_step(self, batch, batch_idx):
        paths = batch
        u, v, a, a_exp, a_real = self.forward(paths)
        dot_ulen = dot(u, u)
        dot_vlen = dot(v, v)
        dot_orth = dot(u, v)
        uz = u[2] * u[2]
        vz = v[2] * v[2]
        ulen_loss = F.mse_loss(dot_ulen, th.ones_like(dot_ulen))
        vlen_loss = F.mse_loss(dot_vlen, th.ones_like(dot_vlen))
        orth_loss = F.mse_loss(dot_orth, th.zeros_like(dot_orth))
        uz_loss = F.mse_loss(uz, th.zeros_like(uz))
        vz_loss = F.mse_loss(vz, th.zeros_like(vz))
        asgn_loss = F.mse_loss(a_real, a_exp)

        a = th.reshape(a, [1, 1, 721, 1504, 1])
        a = th.concatenate([a, a], dim=-1)
        lapl_loss = F.mse_loss(a * 2, sph.laplacian(a))

        loss = ulen_loss + vlen_loss + orth_loss + asgn_loss + uz_loss + vz_loss + lapl_loss
        self.log("lapl_loss", lapl_loss, prog_bar=True, logger=True)
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
        xx, yy = np.arange(1504), np.arange(721)
        return xx, yy, np.random.random([721, 1504]) * 3, np.random.random([721, 1504]) * np.pi * 2

    def __len__(self):
        return 10000


dataset = RandomPathDataset()
train, valid = th.utils.data.random_split(dataset, [9000, 1000])

# -------------------
# Step 3: Train
# -------------------
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    finder = BestFinder()
    trainer = L.Trainer(max_epochs=1)
    train_loader = th.utils.data.DataLoader(train, batch_size=1, num_workers=1, persistent_workers=True)
    valid_loader = th.utils.data.DataLoader(valid, batch_size=1, num_workers=1)

    trainer.fit(finder, train_loader, valid_loader)

    th.save(finder.a, 'a.dat')
    th.save(finder.ux, 'ux.dat')
    th.save(finder.uy, 'uy.dat')
    th.save(finder.vx, 'vx.dat')
    th.save(finder.vy, 'vy.dat')

