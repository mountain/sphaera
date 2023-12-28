import torch as th

from sphaera.plot.plot3d import plot_scalar


def strip(data):
    d = data.reshape(1, 1, 181, 376, 1)
    return d[:, :, :, 8:368, 0:1]


with th.no_grad():
    a = th.load('examples/a.dat')
    plot_scalar('a', strip(a))
