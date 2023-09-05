#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import get_grid_search_hyperparameters
from argparse import Namespace
import pytest

DEFAULT_HYPERPARAMETERS = {
    "vect__ngram_range": [(1, 2), (1, 4)],
    "vect__min_df": (0.1, 0.2, 0.3),
    "clf__max_depth": (10, 15),
    "clf__n_estimators": (100, 200),
    "clf__min_samples_leaf": (2, 3),
}

DEBUG_HYPERPARAMETERS = {"vect__ngram_range": [(1, 2)]}


@pytest.mark.parametrize(
    "args, hyperparameters",
    [
        (Namespace(debug=False, cv_splits=5), DEFAULT_HYPERPARAMETERS),
        (Namespace(debug=True, cv_splits=5), DEBUG_HYPERPARAMETERS),
    ],
)
def test_get_grid_search_paramters(args, hyperparameters, monkeypatch, request):
    monkeypatch.setattr("ipdb.set_trace", lambda: None)
    assert get_grid_search_hyperparameters(args) == hyperparameters
    if args.debug:
        assert args.cv_splits == 2
    else:
        assert args.cv_splits == 5
