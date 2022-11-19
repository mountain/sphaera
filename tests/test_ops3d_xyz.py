# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th
import sphaera as sph

from torch import Tensor
from sphaera.core3d.gridsys.regular3 import RegularGrid


class TestOpsXYZ(unittest.TestCase):

    def setUp(self):
        sph.bind(RegularGrid(
            basis='x,y,z',
            L=51, W=51, H=51,
            east=6.0, west=1.0,
            north=6.0, south=1.0,
            upper=6.0, lower=1.0
        ))
        sph.use('xyz')
        sph.use('theta,phi,r')
        sph.use('thetaphir')

    def tearDown(self):
        sph.clear()

    def magnitude(self, var):
        # return magnitude of variable to test accuracy
        if th.is_tensor(var):
            var = var.cpu().numpy()
        if isinstance(var, np.ndarray):
            return 10 ** int(np.log10(np.abs(var.max())) + 1)
        else:
            return 10 ** int(np.log10(np.abs(var)) + 1)

    def assertAlmostEqualWithMagnitude(self, excepted, test, magnitude=6):
        if (isinstance(excepted, Tensor) and th.abs(excepted).max().cpu().numpy() == 0.0) or \
                (isinstance(excepted, float) and excepted == 0.0):
            return self.assertAlmostEqual(0.0, test.max().cpu().numpy(), magnitude)
        else:
            mag = self.magnitude(excepted)
            err = th.abs(excepted - test) / mag
            return self.assertAlmostEqual(0.0, err.max().cpu().numpy(), magnitude)

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

    def assertAlmostEqualWithMagExceptBoundary(self, excepted, test, magnitude=6):
        if isinstance(excepted, float):
            test = test[:, :, 1:-1, 1:-1, 1:-1]
            return self.assertAlmostEqualWithMagnitude(excepted, test, magnitude)
        else:
            excepted, test = excepted[:, :, 1:-1, 1:-1, 1:-1], test[:, :, 1:-1, 1:-1, 1:-1]
            return self.assertAlmostEqualWithMagnitude(excepted, test, magnitude)

    def test_grad0(self):
        dLx, dLy, dLz = sph.dL

        g0, g1, g2 = sph.grad(sph.x)

        self.assertAlmostEqualWithMagExceptBoundary(5.0, g0 * dLx * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g2, 6)

        g0, g1, g2 = sph.grad(sph.y)

        self.assertAlmostEqualWithMagExceptBoundary(0.0, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(5.0, g1 * dLy * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g2, 6)

        g0, g1, g2 = sph.grad(sph.z)

        self.assertAlmostEqualWithMagExceptBoundary(0.0, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(5.0, g2 * dLz * (sph.H - 1), 6)

    def test_grad1(self):
        # f = x * y * y + z * z * z
        # expect that grad_f = y * y, 2 * x * y, 3 * z * z

        fld = sph.x * sph.y * sph.y + sph.z * sph.z * sph.z
        e0, e1, e2 = sph.y * sph.y, 2 * sph.x * sph.y, 3 * sph.z * sph.z
        g0, g1, g2 = sph.grad(fld)

        self.assertAlmostEqualWithMagExceptBoundary(e0, g0, 3)
        self.assertAlmostEqualWithMagExceptBoundary(e1, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(e2, g2, 4)

    def test_div1(self):
        # F = (0, phi * theta, 0);
        # expect that div_F = theta * (-z * y / (r ** 2 * sqrt(r ** 2 - z ** 2))) + phi * x / (x ** 2 + y ** 2)

        fld = sph.zero, sph.phi * sph.theta, sph.zero
        expected = sph.theta * (-sph.z * sph.y / (sph.r ** 2 * th.sqrt(sph.r ** 2 - sph.z ** 2))) + sph.phi * sph.x / \
            (sph.x ** 2 + sph.y ** 2)
        test = sph.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

    def test_div2(self):
        # F = (-y, x * y, z);
        # expect that div_F = x + 1

        fld = - sph.y, sph.x * sph.y, sph.z
        expected = sph.x + 1
        test = sph.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 6)

    def test_div3(self):
        # F = y * y, 2 * x * y, 3 * z * z
        # expect that div_F = 2 * x + 6 * z

        fld = sph.y * sph.y, 2 * sph.x * sph.y, 3 * sph.z * sph.z
        expected = 2 * sph.x + 6 * sph.z
        test = sph.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 6)

    def test_lapacian(self):
        # f = x * y * y + z * z * z
        # expect that grad_f = y * y, 2 * x * y, 3 * z * z

        fld = sph.x * sph.y * sph.y + sph.z * sph.z * sph.z
        e0, e1, e2 = sph.y * sph.y, 2 * sph.x * sph.y, 3 * sph.z * sph.z
        g0, g1, g2 = sph.grad(fld)

        self.assertAlmostEqualWithMagExceptBoundary(e0, g0, 3)
        self.assertAlmostEqualWithMagExceptBoundary(e1, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(e2, g2, 4)

        # F = y * y, 2 * x * y, 3 * z * z = grad(f)
        # expect that div_F = 2 * x + 6 * z

        expected = 2 * sph.x + 6 * sph.z
        test = sph.div((g0, g1, g2))

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

        # f = x * y * y + z * z * z
        # expect that lapacian_f = 2 * x + 6 * z

        fld = sph.x * sph.y * sph.y + sph.z * sph.z * sph.z
        expected = 2 * sph.x + 6 * sph.z
        test = sph.laplacian(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

    def test_div_sensibility(self):
        # F = y * y, 2 * x * y, 3 * z * z
        fld = sph.y * sph.y, 2 * sph.x * sph.y, 3 * sph.z * sph.z

        noise = th.randn_like(fld[0]) * th.std(fld[0]) * 1e-6, \
            th.randn_like(fld[1]) * th.std(fld[1]) * 1e-6, \
            th.randn_like(fld[2]) * th.std(fld[2]) * 1e-6
        fld_ = fld[0] + noise[0], fld[1] + noise[1], fld[2] + noise[2]
        self.assertAlmostEqualWithMagExceptBoundary(fld[0], fld_[0], 5)
        self.assertAlmostEqualWithMagExceptBoundary(fld[1], fld_[1], 5)
        self.assertAlmostEqualWithMagExceptBoundary(fld[2], fld_[2], 5)

        div1 = sph.div(fld)
        div2 = sph.div(fld_)
        self.assertAlmostEqualWithMagExceptBoundary(div1, div2, 4)

        diff_div1_0, diff_div1_1, diff_div1_2 = sph.diff(div1)
        diff_div2_0, diff_div2_1, diff_div2_2 = sph.diff(div2)
        self.assertAlmostEqualWithMagExceptBoundary(diff_div1_0, diff_div2_0, 2)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, diff_div2_1, 2)
        self.assertAlmostEqualWithMagExceptBoundary(diff_div1_2, diff_div2_2, 2)

    def test_curl1(self):
        # F = (y, -x, 0)
        # expect that curl_F = (0, 0, -2)

        fld = sph.y, - sph.x, sph.zero
        expt_x, expt_y, expt_z = sph.zero, sph.zero, -2 * sph.one
        test_x, test_y, test_z = sph.curl(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_z, test_z, 6)

    def test_curl2(self):
        # F = (0, - x**2, 0)
        # expect that curl_F = (0, 0, -2 * x)

        fld = sph.zero, - sph.x * sph.x, sph.zero
        expt_x, expt_y, expt_z = sph.zero, sph.zero, -2 * sph.x
        test_x, test_y, test_z = sph.curl(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_z, test_z, 6)

        self.assertAlmostEqualWithMagnitude(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagnitude(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagnitude(expt_z, test_z, 6)

    def test_zero_curl(self):
        g0, g1, g2 = sph.curl(sph.grad(sph.x))

        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.y))

        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.z))

        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.z * sph.z + sph.x * sph.y))

        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g2, 6)

    def test_zero_div(self):
        g = sph.div(sph.curl(sph.thetaphir.phi))
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g, 6)

        g = sph.div(sph.curl(sph.thetaphir.theta))
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g, 6)

        g = sph.div(sph.curl(sph.thetaphir.r))
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g, 6)

        phx, phy, phz = sph.thetaphir.phi
        thx, thy, thz = sph.thetaphir.theta
        rx, ry, rz = sph.thetaphir.r
        g = sph.div(sph.curl((rx * rx + phx * thx, ry * ry + phy * thy, rz * rz + phz * thz)))
        self.assertAlmostEqualWithMagExceptBoundary(sph.zero, g, 6)
