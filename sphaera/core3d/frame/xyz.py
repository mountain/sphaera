# -*- coding: utf-8 -*-

import sphaera as iza
import torch as th

from cached_property import cached_property
from torch import Tensor
from typing import Tuple


mps_ready = th.backends.mps.is_available() and th.backends.mps.is_built()
cuda_ready = th.cuda.is_available()


class XYZFrame:
    def __init__(self, grid):
        self.grid = grid
        self.default_device = -1
        iza.use('x,y,z')

    def get_device(self):
        return self.default_device

    def set_device(self, ix):
        self.default_device = ix
        if cuda_ready:
            self.grid.cuda(device=ix)
        elif mps_ready:
            self.grid = self.grid.to(th.device("mps"))

    @cached_property
    def x(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.one, self.grid.zero, self.grid.zero

    @cached_property
    def y(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.zero, self.grid.one, self.grid.zero

    @cached_property
    def z(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.zero, self.grid.zero, self.grid.one


_name_ = 'xyz'
_clazz_ = XYZFrame
