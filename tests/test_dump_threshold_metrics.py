#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.train import dump_threshold_metrics
import pytest
import json
import os


@pytest.mark.parametrize("threshold, precision, recall", [(0.75, 0.99, 0.82)])
def test_dump_threshold_metrics(threshold, precision, recall, tmpdir):
    dump_threshold_metrics(tmpdir, threshold, precision, recall)
    assert os.path.exists(os.path.join(tmpdir, "threshold_metrics.json"))
    with open(os.path.join(tmpdir,
                           "threshold_metrics.json")) as input_file_stream:
        threshold_metrics = json.load(input_file_stream)
    assert {
        "threshold": threshold,
        "precision": precision,
        "recall": recall
    } == threshold_metrics
