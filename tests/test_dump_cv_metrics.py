#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.train import dump_cv_metrics
import numpy as np
import pytest
import json
import os


class MockGridSearchCV:
    def __init__(self, input_dict):
        self.cv_results_ = input_dict


@pytest.mark.parametrize("gs_policy_clf", [
    MockGridSearchCV({
        "a": [1, 2, 3],
        "b": [1, 2, 3]
    }),
    MockGridSearchCV({
        "a": ["a", "b", "c"],
        "b": ["a", "b", "c"]
    }),
    MockGridSearchCV({
        "a": np.array([1, 2, 3]),
        "b": np.array([4, 5, 6])
    }),
    MockGridSearchCV({
        "a": [1, 2, 3],
        "b": np.array([4, 5, 6])
    })
])
def test_dump_args(gs_policy_clf, tmpdir):
    dump_cv_metrics(tmpdir, gs_policy_clf)
    assert os.path.exists(os.path.join(tmpdir, "cv_metrics.json"))
    with open(os.path.join(tmpdir, "cv_metrics.json")) as input_file_stream:
        cv_metrics = json.load(input_file_stream)
    assert {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in gs_policy_clf.cv_results_.items()
    } == cv_metrics
