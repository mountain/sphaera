# -*- coding: utf-8 -*-

import torch as th
import sphaera as sph


_name_ = 'theta,phi,r'
_params_ = ('theta', 'phi', 'r')


def transform(theta, phi, r, **kwargs):
    phi, theta, r = sph.align(phi), sph.align(theta), sph.align(r)
    x = r * th.cos(phi) * th.cos(theta)
    y = r * th.cos(phi) * th.sin(theta)
    z = r * th.sin(phi)
    return x, y, z


def dtransform(theta, phi, r, dtheta, dphi, dr, **kwargs):
    phi, theta, r = sph.align(phi), sph.align(theta), sph.align(r)
    dphi, dtheta, dr = sph.align(dphi), sph.align(dtheta), sph.align(dr)
    dx = - r * th.sin(phi) * th.cos(theta) * dphi - r * th.cos(phi) * th.sin(theta) * dtheta + th.cos(phi) * th.cos(theta) * dr
    dy = - r * th.sin(phi) * th.sin(theta) * dphi + r * th.cos(phi) * th.cos(theta) * dtheta + th.cos(phi) * th.sin(theta) * dr
    dz = r * th.cos(phi) * dphi + th.sin(phi) * dr

    return dx, dy, dz


def inverse(x, y, z, **kwargs):
    x, y, z = sph.align(x), sph.align(y), sph.align(z)
    r = th.sqrt(x * x + y * y + z * z)
    theta = th.atan2(y, x)
    phi = th.asin(z / r)
    return theta, phi, r


def dinverse(x, y, z, dx, dy, dz, **kwargs):
    x, y, z = sph.align(x), sph.align(y), sph.align(z)
    dx, dy, dz = sph.align(dx), sph.align(dy), sph.align(dz)

    r = th.sqrt(x * x + y * y + z * z)

    dr = (x * dx + y * dy + z * dz) / r
    dtheta = (x * dy - y * dx) / (x * x + y * y)
    dphi = (dz - z / r * dr) / th.sqrt(x * x + y * y)

    return dtheta, dphi, dr

