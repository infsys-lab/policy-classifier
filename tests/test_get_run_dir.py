#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.train import get_run_dir
import os


def test_get_run_dir():
    assert os.path.join(os.path.dirname(os.path.dirname(get_run_dir())),
                        "tests") == os.path.dirname(__file__)
