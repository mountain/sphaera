import torch as th
import numpy as np
import xarray as xr

import sphaera as sph


if sph.mps_ready or sph.cuda_ready:
    sph.set_device(0)
else:
    sph.set_device(-1)

from sphaera.core3d.gridsys.regular3 import RegularGrid
from sphaera.core3d.vec3 import dot, norm, cross, normalize, mult
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
    # plot_scalar('wind-velocity', strip(velocity))

    r_0 = sph.thetaphir.r[0][:, :, :, :, 0:1]
    r_1 = sph.thetaphir.r[1][:, :, :, :, 0:1]
    r_2 = sph.thetaphir.r[2][:, :, :, :, 0:1]
    r = (r_0, r_1, r_2)
    r = normalize(r)
    # plot_scalar('r', strip(norm(r)))

    spectrum = th.zeros_like(u10)
    for ix in range(32, 1472, 1):
        print(ix)
        for jx in range(721):
            axis = r_0[:, :, jx, ix, 0:1], r_1[:, :, jx, ix, 0:1], r_2[:, :, jx, ix, 0:1]
            signature = th.sign(dot(r, axis))
            frame = cross(r, axis)
            frame = mult(normalize(frame), (signature, signature, signature))
            val = th.sum(dot(frame, wnd))
            spectrum[0, 0, jx, ix, 0] = val

    th.save(spectrum, 'spectrum.dat')
    spectrum = th.log(spectrum)
    plot_scalar('spectrum', strip(spectrum))
