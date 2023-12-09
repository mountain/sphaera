import torch as th

from sphaera.plot.plot3d import plot_scalar


def strip(data):
    d = data.reshape(1, 1, 721, 1504, 1)
    return d[:, :, :, 32:1472, 0:1]


with th.no_grad():
    velocity = th.load('examples/velocity.dat')
    plot_scalar('velocity', strip(velocity))
    spectrum = th.load('examples/spectrum.dat')
    plot_scalar('spectrum', strip(spectrum))
