#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import dump_metrics
import pytest
import json
import os


@pytest.mark.parametrize(
    "metrics",
    [{"a": {"b": 1, "c": 2, "d": 3}, "e": {"f": 1}, "g": {"h": 1, "i": 2, "j": 3}}],
)
def test_dump_metrics(metrics, tmpdir):
    dump_metrics(tmpdir, metrics)
    assert os.path.exists(os.path.join(tmpdir, "metrics.json"))
    with open(os.path.join(tmpdir, "metrics.json")) as input_file_stream:
        metrics_reload = json.load(input_file_stream)
    assert metrics == metrics_reload
