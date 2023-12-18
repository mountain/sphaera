import torch as th
import numpy as np
import xarray as xr

import sphaera as sph

if sph.mps_ready or sph.cuda_ready:
    sph.set_device(0)
else:
    sph.set_device(-1)


from sphaera.core3d.gridsys.regular3 import RegularGrid
from sphaera.core3d.vec3 import norm
from sphaera.plot.plot3d import plot_scalar

wind = xr.open_dataset('examples/wind.nc')


def cast(data):
    d = np.array(data, dtype=np.float64).reshape(721, 1440)
    return sph.cast(np.concatenate((d[:, 1439:1407:-1], d, d[:, 0:32:1]), axis=1)).reshape(1, 1, 721, 1504, 1)


def strip(data):
    d = data.reshape(1, 1, 721, 1504, 1)
    return d[:, :, :, 32:1472, 0:1]


with th.no_grad():
    # setup grid system
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

    u10 = cast(wind['u10'].data)
    v10 = cast(wind['v10'].data)
    wnd = (u10, v10, sph.zero)
    velocity = norm(wnd)
    th.save(velocity, 'velocity.dat')
    plot_scalar('wind-velocity', strip(velocity))
