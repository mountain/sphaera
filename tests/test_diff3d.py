# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th

import sphaera as sph

from sphaera.core3d.diffr.regular3 import central, sobel3, sobel5, sharr3, centralx, centraly, centralz
from sphaera.core3d.gridsys.regular3 import RegularGrid


class TestDiff(unittest.TestCase):

    def setUp(self):
        sph.bind(RegularGrid(
            basis='lng,lat,alt',
            W=51, L=51, H=51,
            east=119.0, west=114.0,
            north=42.3, south=37.3,
            upper=16000.0, lower=0.0
        ))

    def tearDown(self):
        sph.clear()

    def magnitude(self, var):
        # return magnitude of variable to test accuracy
        if th.is_tensor(var):
            var = var.cpu().numpy()
        if isinstance(var, np.ndarray):
            return 10 ** int(np.log10(np.abs(var.max()) + 1))
        else:
            return 10 ** int(np.log10(np.abs(var) + 1))

    def assertAlmostEqualWithMagnitude(self, excepted, test, magnitude=6):
        m = self.magnitude(excepted)
        return self.assertAlmostEqual(excepted / m, test / m, magnitude)

    def test_order(self):
        kx = centralx[0, 0].cpu().numpy()
        self.assertAlmostEqual(1.0, np.max(kx[2:3, :, :] - kx[0:1, :, :]))
        self.assertAlmostEqual(0, np.max(kx[1:2, :, :]))

        ky = centraly[0, 0].cpu().numpy()
        self.assertAlmostEqual(1.0, np.max(ky[:, 2:3, :] - ky[:, 0:1, :]))
        self.assertAlmostEqual(0, np.max(ky[:, 1:2, :]))

        kz = centralz[0, 0].cpu().numpy()
        self.assertAlmostEqual(1.0, np.max(kz[:, :, 2:3] - kz[:, :, 0:1]))
        self.assertAlmostEqual(0, np.max(kz[:, :, 1:2]))

    def test_central(self):
        gx, gy, gz = central(sph.lng)

        self.assertAlmostEqualWithMagnitude(5.0, gx.mean().cpu().numpy() * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = central(sph.lat)

        self.assertAlmostEqualWithMagnitude(0.0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(5.0, gy.mean().cpu().numpy() * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = central(sph.alt)

        self.assertAlmostEqualWithMagnitude(0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(16000, gz.mean().cpu().numpy() * (sph.H - 1), 6)

    def test_sobel3(self):
        gx, gy, gz = sobel3(sph.lng)

        self.assertAlmostEqualWithMagnitude(5.0, gx.mean().cpu().numpy() * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sobel3(sph.lat)

        self.assertAlmostEqualWithMagnitude(0.0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(5.0, gy.mean().cpu().numpy() * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sobel3(sph.alt)

        self.assertAlmostEqualWithMagnitude(0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(16000, gz.mean().cpu().numpy() * (sph.H - 1), 6)

    def test_sobel5(self):
        gx, gy, gz = sobel5(sph.lng)

        self.assertAlmostEqualWithMagnitude(5.0, gx.mean().cpu().numpy() * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sobel5(sph.lat)

        self.assertAlmostEqualWithMagnitude(0.0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(5.0, gy.mean().cpu().numpy() * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sobel5(sph.alt)

        self.assertAlmostEqualWithMagnitude(0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(16000, gz.mean().cpu().numpy() * (sph.H - 1), 6)

    def test_sharr3(self):
        gx, gy, gz = sharr3(sph.lng)

        self.assertAlmostEqualWithMagnitude(5.0, gx.mean().cpu().numpy() * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sharr3(sph.lat)

        self.assertAlmostEqualWithMagnitude(0.0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(5.0, gy.mean().cpu().numpy() * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, gz.mean().cpu().numpy(), 6)

        gx, gy, gz = sharr3(sph.alt)

        self.assertAlmostEqualWithMagnitude(0, gx.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(0, gy.mean().cpu().numpy(), 6)
        self.assertAlmostEqualWithMagnitude(16000, gz.mean().cpu().numpy() * (sph.H - 1), 6)
