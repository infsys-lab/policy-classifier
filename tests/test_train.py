#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.train import preprocess, get_closest_value_index
import numpy as np


def test_preprocess():
    assert preprocess("") == ""
    assert preprocess("my website is https://example.com") == "my website is "
    assert preprocess("my website is http://example.com") == "my website is "
    assert preprocess("my email is person@example.com") == "my email is person"
    assert preprocess(".%&$") == "    "
    assert preprocess("my email is person @ example dot com"
                      ) == "my email is person at example dot com"
    assert preprocess("what a\nnice day") == "what a nice day"


def test_get_closest_value_index():
    assert get_closest_value_index(np.array([1., 2., 3.]), 1.) == 0
    assert get_closest_value_index(np.array([1., 2., 3.]), 2.) == 1
    assert get_closest_value_index(np.array([1., 2., 3.]), 3.) == 2
    assert get_closest_value_index(np.array([1., 2., 3.]), 1.5) == 0
    assert get_closest_value_index(np.array([1., 2., 3.]), 2.5) == 1
    assert get_closest_value_index(np.array([1., 2., 3.]), 3.5) == 2
    assert get_closest_value_index(np.array([1., 1., 2.]), 1) == 0
    assert get_closest_value_index(np.array([1., 2., 1.]), 1) == 0
    assert get_closest_value_index(np.array([1., 1., 1.]), 1) == 0
