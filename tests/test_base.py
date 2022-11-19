# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th
import sphaera as sph

from sphaera.core3d.vec3 import add, mult, normalize
from sphaera.core3d.gridsys.regular3 import RegularGrid


class TestBase(unittest.TestCase):

    def setUp(self):
        sph.bind(RegularGrid(
            basis='lng,lat,alt',
            W=11, L=20, H=2,
            east=-180, west=180,
            north=89.99, south=-89.99,
            upper=1.0, lower=0.0
        ))
        sph.use('x,y,z')
        sph.use('xyz')
        sph.use('theta,phi,r')
        sph.use('thetaphir')

    def tearDown(self):
        sph.clear()

    def test_elements(self):
        dl1, dl2, dl3 = sph.dL
        self.assertGreater(th.min(dl1).detach().numpy(), 0)
        self.assertGreater(th.min(dl2).detach().numpy(), 0)
        self.assertGreater(th.min(dl3).detach().numpy(), 0)
        ds1, ds2, ds3 = sph.dS
        self.assertGreater(th.min(ds1).detach().numpy(), 0)
        self.assertGreater(th.min(ds2).detach().numpy(), 0)
        self.assertGreater(th.min(ds3).detach().numpy(), 0)
        dv = sph.dV
        self.assertGreater(th.min(dv).detach().numpy(), 0)
