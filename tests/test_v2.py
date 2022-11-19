# -*- coding: utf-8 -*-

import unittest

import numpy as np

import sphaera as sph


class TestV2(unittest.TestCase):

    def setUp(self):
        self.zero = sph.cast(np.zeros([1, 1, 2, 2]))
        self.one = sph.cast(np.ones([1, 1, 2, 2]))

    def tearDown(self):
        pass
