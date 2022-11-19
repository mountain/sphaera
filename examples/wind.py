import numpy as np
import torch as th
import sphaera as sph
import xarray as xr

from sphaera.core3d.gridsys.regular3 import RegularGrid
from sphaera.core3d.vec3 import mult, div, add, normalize, norm
from sphaera.plot.plot3d import plot_scalar, plot_vector

wind = xr.open_dataset('examples/wind.nc')


def cast(data):
    d = data.reshape(721, 1440)
    return sph.cast(np.concatenate((d[:, 1439:1407:-1], d, d[:, 0:32:1]), axis=1)).reshape(1, 1, 721, 1504, 1)


def strip(data):
    d = data.reshape(1, 1, 721, 1504, 2)
    return d[:, :, :, 32:1472, 0:1]


sph.default_device = -1

# setup grid system
sph.bind(RegularGrid(
    basis='lng,lat,alt',
    W=1504, L=721, H=2,
    east=0 - 0.25 * 32, west=360 + 0.25 * 32,
    north=-89.999, south=89.999,
    upper=11.0, lower=10.0
))
sph.use('x,y,z')
sph.use('xyz')
sph.use('theta,phi,r')
sph.use('thetaphir')

u10 = cast(wind['u10'].data)
v10 = cast(wind['v10'].data)
u10 = th.cat((u10, u10), dim=4)
v10 = th.cat((v10, v10), dim=4)
wnd = (u10, v10, sph.zero)
plot_scalar('wind velocity', strip(norm(wnd)))

# vortex = sph.curl(wnd)
# plot_scalar('length', strip(th.log(norm(vortex))))

plot_scalar('theta', strip(sph.theta))
plot_scalar('phi', strip(sph.phi))
plot_scalar('Theta', strip(norm(sph.thetaphir.theta)))
plot_scalar('Phi', strip(norm(sph.thetaphir.phi)))

u = mult(sph.thetaphir.theta, (np.sin(sph.phi), np.sin(sph.phi), np.sin(sph.phi)))
plot_scalar('u', strip(norm(u)))
plot_scalar('curl u', strip(norm(sph.curl(u))))

alpha_p = np.pi / 4
alpha_q = np.pi / 2

a = np.cos(alpha_p) * np.cos(sph.phi) + np.sin(alpha_p) * (np.sin(sph.theta) * sph.zero - np.cos(sph.theta) * np.sin(sph.phi))
b = np.cos(alpha_p) * sph.zero + np.sin(alpha_p) * (np.sin(sph.theta) * sph.one + np.cos(sph.theta) * sph.zero)
p = add(mult(sph.thetaphir.theta, (a, a, a)), mult(sph.thetaphir.phi, (b, b, b)))

a = np.cos(alpha_q) * np.cos(sph.phi) + np.sin(alpha_q) * np.sin(sph.phi) * np.cos(sph.theta)
b = np.sin(alpha_q) * np.sin(sph.theta)
q = add(mult(sph.thetaphir.theta, (a, a, a)), mult(sph.thetaphir.phi, (b, b, b)))

plot_scalar('p', strip(norm(p)))
plot_scalar('q', strip(norm(q)))

vp = sph.curl(p)
vq = sph.curl(q)

plot_scalar('vp', strip(norm(vp)))
plot_scalar('vq', strip(norm(vq)))

print(th.sum(sph.dot(vp, vp) * sph.dV))
print(th.sum(sph.dot(vq, vq) * sph.dV))
print(th.sum(sph.dot(vp, vq) * sph.dV))

