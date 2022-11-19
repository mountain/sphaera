# -*- coding: utf-8 -*-

import unittest

import numpy as np
import sphaera as sph

from sphaera.core3d.vec3 import box
from sphaera.core3d.gridsys.regular3 import RegularGrid


class TestFrame(unittest.TestCase):

    def setUp(self):
        sph.bind(RegularGrid(
            basis='lng,lat,alt',
            W=51, L=51, H=51,
            east=119.0, west=114.0,
            north=42.3, south=37.3,
            upper=16000.0, lower=0.0
        ))
        sph.use('thetaphir')
        sph.use('xyz')
        sph.use('x,y,z')

    def tearDown(self):
        sph.clear()

    def test_basis_ortho(self):
        phx, phy, phz = sph.thetaphir.phi
        phx = phx.cpu().numpy()
        phy = phy.cpu().numpy()
        phz = phz.cpu().numpy()
        self.assertAlmostEqual(1, np.max(phx * phx + phy * phy + phz * phz))
        self.assertAlmostEqual(1, np.min(phx * phx + phy * phy + phz * phz))

        thx, thy, thz = sph.thetaphir.theta
        thx = thx.cpu().numpy()
        thy = thy.cpu().numpy()
        thz = thz.cpu().numpy()
        val = thx * thx + thy * thy + thz * thz
        self.assertAlmostEqual(1, np.max(val))
        self.assertAlmostEqual(1, np.min(val))

        rx, ry, rz = sph.thetaphir.r
        rx = rx.cpu().numpy()
        ry = ry.cpu().numpy()
        rz = rz.cpu().numpy()
        self.assertAlmostEqual(1, np.max(rx * rx + ry * ry + rz * rz))
        self.assertAlmostEqual(1, np.min(rx * rx + ry * ry + rz * rz))

        self.assertAlmostEqual(0, np.max(phx * thx + phy * thy + phz * thz))
        self.assertAlmostEqual(0, np.min(phx * rx + phy * ry + phz * rz))
        self.assertAlmostEqual(0, np.min(thx * rx + thy * ry + thz * rz))

    def test_basis_righthand(self):
        d = box(sph.thetaphir.theta, sph.thetaphir.phi, sph.thetaphir.r).cpu().numpy()
        self.assertAlmostEqual(1, np.mean(d))
        self.assertAlmostEqual(1, np.min(d))
        self.assertAlmostEqual(1, np.max(d))

        d = box(sph.xyz.x, sph.xyz.y, sph.xyz.z).cpu().numpy()
        self.assertAlmostEqual(1, np.mean(d))
        self.assertAlmostEqual(1, np.min(d))
        self.assertAlmostEqual(1, np.max(d))

    def test_basis_diff(self):
        from sphaera.core3d.basis.theta_phi_r import transform

        x, y, z = transform(sph.theta + 0.0001, sph.phi, sph.r)
        thx, thy, thz = sph.normalize((x - sph.x, y - sph.y, z - sph.z))
        a = sph.dot((thx, thy, thz), sph.thetaphir.theta)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))

        x, y, z = transform(sph.theta, sph.phi + 0.0001, sph.r)
        phx, phy, phz = sph.normalize((x - sph.x, y - sph.y, z - sph.z))
        a = sph.dot((phx, phy, phz), sph.thetaphir.phi)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))

        x, y, z = transform(sph.theta, sph.phi, sph.r + 0.0001)
        rx, ry, rz = sph.normalize((x - sph.x, y - sph.y, z - sph.z))
        a = sph.dot((rx, ry, rz), sph.thetaphir.r)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))



