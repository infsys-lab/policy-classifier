#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from operator import attrgetter

import pytest

from parser import get_train_parser
from train import main


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_reproducibility(seed, get_dummy_data, monkeypatch, tmpdir):
    def mock_get_first_run_dir(*args, **kwargs):
        return os.path.join(tmpdir, "run_1")

    def mock_get_second_run_dir(*args, **kwargs):
        return os.path.join(tmpdir, "run_2")

    def mock_get_raw_data(*args, **kwargs):
        return attrgetter("X", "y")(get_dummy_data(seed=seed))

    def mock_add_file_handler(*args, **kwargs):
        return None

    monkeypatch.setattr("ipdb.set_trace", lambda: None)
    monkeypatch.setattr("parser.file_path", lambda path: path)
    monkeypatch.setattr("train.get_run_dir", mock_get_first_run_dir)
    monkeypatch.setattr("train.get_raw_data", mock_get_raw_data)
    monkeypatch.setattr("train.add_file_handler", mock_add_file_handler)
    parser = get_train_parser()
    args = parser.parse_args(["--debug"])
    main(args)
    monkeypatch.setattr("train.get_run_dir", mock_get_second_run_dir)
    main(args)
    with open(
        os.path.join(mock_get_first_run_dir(), "metrics.json")
    ) as input_file_stream:
        metrics_first = json.load(input_file_stream)
    with open(
        os.path.join(mock_get_second_run_dir(), "metrics.json")
    ) as input_file_stream:
        metrics_second = json.load(input_file_stream)
    assert metrics_first["clf_report"] == metrics_second["clf_report"]
    assert metrics_first["threshold_metrics"] == metrics_second["threshold_metrics"]
