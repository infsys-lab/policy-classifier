#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import Namespace

import pandas as pd
import pytest

from train import get_raw_data


@pytest.fixture
def args():
    return Namespace(policies_csv="some/test/path")


@pytest.fixture
def mock_df(monkeypatch):
    def mock_df_inner(*args, **kwargs):
        return pd.DataFrame(
            data={
                "policy_text": ["testing once", "testing twice"],
                "is_policy": [False, True],
            }
        )

    monkeypatch.setattr(pd, "read_csv", mock_df_inner)


def test_get_raw_data(args, mock_df):
    X, y = get_raw_data(args)
    assert X == ["testing once", "testing twice"]
    assert y == [0, 1]
