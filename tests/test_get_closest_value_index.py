#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import get_closest_value_index
import numpy as np
import pytest


@pytest.mark.parametrize("vector, closest_value, index",
                         [([1., 2., 3.], 1., 0), ([1., 2., 3.], 2., 1),
                          ([1., 2., 3.], 3., 2), ([1., 2., 3.], 1.5, 0),
                          ([1., 2., 3.], 2.5, 1), ([1., 2., 3.], 3.5, 2),
                          ([1., 1., 2.], 1., 0), ([1., 2., 1.], 1., 0),
                          ([1., 1., 1.], 1., 0)])
def test_get_closest_value_index(vector, closest_value, index):
    assert get_closest_value_index(np.array(vector), closest_value) == index
