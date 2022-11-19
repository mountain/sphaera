# -*- coding: utf-8 -*-

import torch as th
import sphaera as sph

from cached_property import cached_property
from torch import Tensor
from typing import Tuple


class Elements:
    def __init__(self, **kwargs):
        self.default_device = -1

        if sph._grid_.basis == 'theta,phi,r':
            self.dL1 = th.abs(sph.r * th.cos(sph.phi) * sph.dtheta)
            self.dL2 = th.abs(sph.r * sph.dphi)
            self.dL3 = th.abs(sph.dr)
        if sph._grid_.basis == 'x,y,z':
            self.dL1 = th.abs(sph.dx)
            self.dL2 = th.abs(sph.dy)
            self.dL3 = th.abs(sph.dz)
        if sph._grid_.basis == 'lng,lat,alt':
            sph.use('theta,phi,r', **kwargs)
            self.dL1 = th.abs(sph.r * th.cos(sph.phi) * sph.dtheta)
            self.dL2 = th.abs(sph.r * sph.dphi)
            self.dL3 = th.abs(sph.dr)

        self.dS3 = self.dL1 * self.dL2
        self.dS1 = self.dL3 * self.dL2
        self.dS2 = self.dL3 * self.dL1

        self.dVol = self.dL1 * self.dL2 * self.dL3

    def get_device(self):
        return self.default_device

    def set_device(self, ix):
        if ix >= 0:
            self.dL1 = self.dL1.cuda(device=ix)
            self.dL2 = self.dL2.cuda(device=ix)
            self.dL3 = self.dL3.cuda(device=ix)
            self.dS1 = self.dS1.cuda(device=ix)
            self.dS2 = self.dS2.cuda(device=ix)
            self.dS3 = self.dS3.cuda(device=ix)
            self.dVol = self.dVol.cuda(device=ix)
        self.default_device = ix

    @cached_property
    def dL(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.dL1, self.dL2, self.dL3

    @cached_property
    def dS(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.dS1, self.dS2, self.dS3

    @cached_property
    def dV(self) -> Tensor:
        return self.dVol
