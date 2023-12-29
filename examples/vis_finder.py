import torch as th
import sphaera as sph

from sphaera.plot.plot3d import plot_scalar
from sphaera.core3d.gridsys.regular3 import RegularGrid


def strip(data):
    d = data.reshape(1, 1, 721, 1504, -1)
    return d[:, :, :, 32:1472, 0:1]


with th.no_grad():
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

    ux = th.load('examples/ux.dat')
    plot_scalar('ux', strip(ux))

    uy = th.load('examples/uy.dat')
    plot_scalar('uy', strip(uy))

    a = th.load('examples/a.dat')
    a = th.reshape(a, [1, 1, 721, 1504, -1])
    a = th.concatenate([a, a], dim=-1).double()

    plot_scalar('a', strip(a))

    b = sph.laplacian(a)
    plot_scalar('b', strip(b))
