import torch as th
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import xarray as xr
import sphaera as sph

from sphaera.core3d.gridsys.regular3 import RegularGrid
from sphaera.core3d.vec3 import dot, norm, cross, normalize, mult, sub, scale

wind = xr.open_dataset('examples/wind.nc')


def cast(data):
    d = np.array(data, dtype=np.float64).reshape(721, 1440)
    return sph.cast(np.concatenate((d[:, 1439:1407:-1], d, d[:, 0:32:1]), axis=1), device=0).reshape(1, 1, 721, 1504, 1)


def strip(data):
    d = data.reshape(1, 1, 721, 1504, 1)
    return d[:, :, :, 32:1472, 0:1]


with th.no_grad():
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
    wnd = (u10, v10, sph.align(sph.zero))
    velocity = norm(wnd)
    th.save(velocity, 'velocity.dat')
    # plot_scalar('wind-velocity', strip(velocity))

    r_0 = sph.align(sph.thetaphir.r[0][:, :, :, :, 0:1])
    r_1 = sph.align(sph.thetaphir.r[1][:, :, :, :, 0:1])
    r_2 = sph.align(sph.thetaphir.r[2][:, :, :, :, 0:1])
    r = (r_0, r_1, r_2)
    r = normalize(r)
    # plot_scalar('r', strip(norm(r)))

    spectrum = th.zeros_like(velocity)
    for ix in range(1504):
        for jx in range(721):
            print(ix, jx)
            axis = r_0[:, :, jx, ix, 0:1], r_1[:, :, jx, ix, 0:1], r_2[:, :, jx, ix, 0:1]
            coeff = dot(r, axis)
            perp = sub(r, scale(axis, coeff))
            frame = cross(axis, perp)
            signature = th.sign(dot(r, axis))
            frame = scale(frame, signature)
            spectrum[0:1, 0:1, jx, ix, 0:1] = th.sum(dot(frame, wnd) * sph._element_.dV) / th.sum(sph._element_.dV)

    th.save(spectrum.to(th.device('cpu')), './examples/spectrum.dat')
