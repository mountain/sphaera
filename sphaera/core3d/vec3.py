# -*- coding: utf-8 -*-

import torch as th


@th.compile
def add(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 + y1, x2 + y2, x3 + y3


@th.compile
def sub(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 - y1, x2 - y2, x3 - y3


@th.compile
def mult(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 * y1, x2 * y2, x3 * y3


@th.compile
def div(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 / y1, x2 / y2, x3 / y3


@th.compile
def dot(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 * y1 + x2 * y2 + x3 * y3


@th.compile
def cross(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return (
        x2 * y3 - x3 * y2,
        x3 * y1 - x1 * y3,
        x1 * y2 - x2 * y1,
    )


@th.compile
def norm(v):
    return th.sqrt(th.sum(dot(v, v), dim=1, keepdim=True))


@th.compile
def normsq(v):
    return th.sum(dot(v, v), dim=1, keepdim=True)


@th.compile
def normalize(v):
    x, y, z = v
    r = th.sqrt(x * x + y * y + z * z)

    return x / r, y / r, z / r


@th.compile
def box(a, b, c):
    return dot(a, cross(b, c))


@th.compile
def det(transform):
    return box(transform[0:1, 0:3], transform[1:2, 0:3], transform[2:3, 0:3])
