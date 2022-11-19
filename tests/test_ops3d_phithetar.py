# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th

import sphaera as sph

from torch import Tensor
from sphaera.core3d.gridsys.regular3 import RegularGrid


class TestOpsPhiThetaR(unittest.TestCase):

    def setUp(self):
        sph.bind(RegularGrid(
            basis='lng,lat,alt',
            L=51, W=51, H=51,
            east=119.0, west=114.0,
            north=42.3, south=37.3,
            upper=16000.0, lower=0.0
        ), r0=6371000)
        sph.use('theta,phi,r')
        sph.use('thetaphir')

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

    def assertAlmostEqualWithMagnitude(self, excepted, test, magnitude=6):
        if isinstance(excepted, Tensor) and th.abs(excepted).max().cpu().numpy() == 0.0 or isinstance(excepted, float) and excepted == 0.0:
            return self.assertAlmostEqual(0.0, test.max().cpu().numpy(), magnitude)
        else:
            mag = self.magnitude(excepted)
            err = th.abs(excepted - test) / mag
            return self.assertAlmostEqual(0.0, err.max().cpu().numpy(), magnitude)

    def test_grad(self):
        dLth, dLph, dLr = sph.dL

        g0, g1, g2 = sph.grad(sph.lng)

        self.assertAlmostEqualWithMagnitude(5.0, g0 * dLth * (sph.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0.0, g2, 6)

        g0, g1, g2 = sph.grad(sph.lat)

        self.assertAlmostEqualWithMagnitude(0.0, g0, 6)
        self.assertAlmostEqualWithMagnitude(5.0, g1 * dLph * (sph.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, g2, 6)

        g0, g1, g2 = sph.grad(sph.alt)

        self.assertAlmostEqualWithMagnitude(0.0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0.0, g1, 6)
        self.assertAlmostEqualWithMagnitude(16000.0, g2 * dLr * (sph.H - 1), 6)

    def test_zero_curl(self):
        g0, g1, g2 = sph.curl(sph.grad(sph.lat))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.lng))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.alt))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = sph.curl(sph.grad(sph.alt * sph.alt + sph.lng * sph.lat))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

    def test_zero_div(self):
        g = sph.div(sph.curl(sph.thetaphir.phi))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        g = sph.div(sph.curl(sph.thetaphir.theta))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        g = sph.div(sph.curl(sph.thetaphir.r))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        phx, phy, phz = sph.thetaphir.phi
        thx, thy, thz = sph.thetaphir.theta
        rx, ry, rz = sph.thetaphir.r
        g = sph.div(sph.curl((rx * rx + phx * thx, ry * ry + phy * thy, rz * rz + phz * thz)))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

    def test_div1(self):
        # F = (-theta, phi * theta, r);
        # expect that div_F = theta * tan(phi) / r + phi / (r * cos(phi)) + 3

        fld = - sph.theta, sph.phi * sph.theta, sph.r

        expected = sph.theta * th.tan(sph.phi) / sph.r + sph.phi / (sph.r * th.cos(sph.phi)) + 3

        test = sph.div(fld)

        self.assertAlmostEqualWithMagnitude(expected, test, 3)

    def test_div2(self):
        # phi0 = phi[:, :, W//2, L//2, H//2], theta0 = theta[:, :, W//2, L//2, H//2], r0 = r[:, :, W//2, L//2, H//2]
        # F = (-(theta - theta0), (phi - phi0) * (theta-theta0), (r-r0));
        # expect that div_F = (theta - theta0) * tan(phi) / r + (phi - phi0) / (r * cos(phi)) + 3 - 2 * r0 / r

        phi0 = sph.phi[:, :, sph.W // 2, sph.L // 2, sph.H // 2]
        theta0 = sph.theta[:, :, sph.W // 2, sph.L // 2, sph.H // 2]
        r0 = sph.r[:, :, sph.W // 2, sph.L // 2, sph.H // 2]

        Fx = -(sph.theta - theta0)
        Fy = (sph.phi - phi0) * (sph.theta - theta0)
        Fz = (sph.r - r0)

        test = sph.div((Fx, Fy, Fz))

        expected = (sph.theta - theta0) * th.tan(sph.phi) / sph.r +\
                   (sph.phi - phi0) / (sph.r * th.cos(sph.phi)) + 3 - 2 * r0 / sph.r

        self.assertAlmostEqualWithMagnitude(expected, test, 3)

    def test_curl1(self):
        # F = (-theta, phi * theta, r);
        # expect that curl_F = (-phi*theta/r, -theta/r, (theta-phi*theta*tan(phi)+1/cos(phi))/r)

        fld = - sph.theta, sph.phi * sph.theta, sph.r

        expected_ph = -sph.phi * sph.theta / sph.r
        expected_th = -sph.theta / sph.r
        expected_r = (sph.theta - sph.phi * sph.theta * th.tan(sph.phi) + 1 / th.cos(sph.phi)) / sph.r

        test_ph, test_th, test_r = sph.curl(fld)

        self.assertAlmostEqualWithMagnitude(expected_ph, test_ph, 6)
        self.assertAlmostEqualWithMagnitude(expected_th, test_th, 6)
        self.assertAlmostEqualWithMagnitude(expected_r, test_r, 6)

    def test_curl2(self):
        # F = (0, cos(phi), 0)
        # expect that curl = (-cos(phi)/r, 0, -2*sin(phi)/r)
        fld = 0, th.cos(sph.phi), 0

        expected_ph = -th.cos(sph.phi) / sph.r
        expected_th = sph.zero
        expected_r = -2 * th.sin(sph.phi) / sph.r

        test_ph, test_th, test_r = sph.curl(fld)

        self.assertAlmostEqualWithMagnitude(expected_ph, test_ph, 6)
        self.assertAlmostEqualWithMagnitude(expected_th, test_th, 6)
        self.assertAlmostEqualWithMagnitude(expected_r, test_r, 6)
