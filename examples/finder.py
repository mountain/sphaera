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

wind = xr.open_dataset('examples/wind.nc')


def cast(data):
    d = np.array(data, dtype=np.float64).reshape(721, 1440)
    return sph.cast(np.concatenate((d[:, 1439:1407:-1], d, d[:, 0:32:1]), axis=1), device=0).reshape(1, 1, 721, 1504, 1)


def strip(data):
    d = data.reshape(1, 1, 721, 1504, 1)
    return d[:, :, :, 32:1472, 0:1]

# --------------------------------
# Step 1: Setup grid system
# --------------------------------

sph.bind(RegularGrid(
    basis='lng,lat,alt',
    W=1504, L=721, H=2,
    east=0 - 0.25 * 32, west=360 + 0.25 * 32,
    north=-89.999, south=89.999,
    upper=11.0 + 63567523, lower=10.0 + 63567523
))
sph.use('x,y,z')
sph.use('xyz')
sph.use('theta,phi,r')
sph.use('thetaphir')

if sph.mps_ready or sph.cuda_ready:
    sph.set_device(0)
else:
    sph.set_device(-1)

u10 = cast(wind['u10'].data)
v10 = cast(wind['v10'].data)
wnd = (u10, v10, sph.zero)
velocity = sph.align(norm(wnd))
th.save(velocity, 'velocity.dat')
# plot_scalar('wind-velocity', strip(velocity))

r_0 = sph.align(sph.thetaphir.r[0][:, :, :, :, 0:1])
r_1 = sph.align(sph.thetaphir.r[1][:, :, :, :, 0:1])
r_2 = sph.align(sph.thetaphir.r[2][:, :, :, :, 0:1])
r = (r_0, r_1, r_2)
r = normalize(r)
# plot_scalar('r', strip(norm(r)))


# ----------------------------------------------
# Step 2: Define a machine learning model
# ----------------------------------------------

class BestVeloFinder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.base_velo = th.nn.Parameter(sph.align(th.ones_like(velocity)))

    def forward(self, x):
        ix, jx = x
        axis = r_0[:, :, jx, ix, 0:1], r_1[:, :, jx, ix, 0:1], r_2[:, :, jx, ix, 0:1]
        signature = th.sign(dot(r, axis))
        scale = signature * self.base_velo[:, :, jx, ix, 0:1]
        frame = cross(r, axis)
        frame = mult(frame, (scale, scale, scale))
        return frame

    def training_step(self, batch, batch_idx):
        points = batch
        frame = self.forward(points)
        val = th.sum(dot(frame, wnd))
        loss = F.mse_loss(val, th.zeros_like(val))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ----------------------------------------------
# Step 3: Define a dataset
# ----------------------------------------------

class RandomPointDataset(D.dataset.Dataset):
    def __getitem__(self, index):
        return random.sample(range(1504), 1)[0], random.sample(range(721), 1)[0]

    def __len__(self):
        return 1504 * 721


dataset = RandomPointDataset()
train, valid = th.utils.data.random_split(dataset, [1504 * 721 // 7 * 6, 1504 * 721 // 7])

# -------------------
# Step 3: Train
# -------------------
finder = BestVeloFinder().to(th.device('mps'))
trainer = L.Trainer()
trainer.fit(finder, th.utils.data.DataLoader(train, batch_size=1), th.utils.data.DataLoader(valid, batch_size=1))

th.save(finder.base_velo, 'base-velocity.dat')
