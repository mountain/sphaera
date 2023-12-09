# -*- coding: utf-8 -*-

import torch as th
import sphaera as sph

from cached_property import cached_property
from torch import Tensor
from typing import Tuple


mps_ready = th.backends.mps.is_available() and th.backends.mps.is_built()
cuda_ready = th.cuda.is_available()


class ThetaPhiRFrame:
    def __init__(self, grid):
        self.default_device = -1
        self.grid = grid
        sph.use('theta,phi,r')

        self.thx = - th.sin(sph.theta)
        self.thy = th.cos(sph.theta)
        self.thz = - sph.zero

        self.phx = - th.sin(sph.phi) * th.cos(sph.theta)
        self.phy = - th.sin(sph.phi) * th.sin(sph.theta)
        self.phz = th.cos(sph.phi)

        self.rx = th.cos(sph.phi) * th.cos(sph.theta)
        self.ry = th.cos(sph.phi) * th.sin(sph.theta)
        self.rz = th.sin(sph.phi)

    def get_device(self):
        return self.default_device

    def set_device(self, ix):
        self.default_device = ix
        if cuda_ready:
            self.thx = self.thx.cuda(device=ix)
            self.thy = self.thy.cuda(device=ix)
            self.thz = self.thz.cuda(device=ix)
            self.phx = self.phx.cuda(device=ix)
            self.phy = self.phy.cuda(device=ix)
            self.phz = self.phz.cuda(device=ix)
            self.rx = self.rx.cuda(device=ix)
            self.ry = self.ry.cuda(device=ix)
            self.rz = self.rz.cuda(device=ix)
        elif mps_ready:
            self.thx = self.thx.to(th.device("mps"))
            self.thy = self.thy.to(th.device("mps"))
            self.thz = self.thz.to(th.device("mps"))
            self.phx = self.phx.to(th.device("mps"))
            self.phy = self.phy.to(th.device("mps"))
            self.phz = self.phz.to(th.device("mps"))
            self.rx = self.rx.to(th.device("mps"))
            self.ry = self.ry.to(th.device("mps"))
            self.rz = self.rz.to(th.device("mps"))

    @cached_property
    def phi(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.phx, self.phy, self.phz

    @cached_property
    def theta(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.thx, self.thy, self.thz

    @cached_property
    def r(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.rx, self.ry, self.rz


_name_ = 'thetaphir'
_clazz_ = ThetaPhiRFrame
