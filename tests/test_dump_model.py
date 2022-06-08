#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.train import dump_model, preprocess
import pytest
import dill
import os

# define global dill setting
dill.settings["recurse"] = True


@pytest.fixture
def fitted_model(get_dummy_data, request):
    dummy_data = get_dummy_data(seed=request.param)
    policy_clf = Pipeline([
        ("vect", TfidfVectorizer(stop_words="english",
                                 preprocessor=preprocess)),
        ("clf",
         RandomForestClassifier(class_weight="balanced",
                                random_state=request.param)),
    ])
    policy_clf.fit(dummy_data.X, dummy_data.y)
    return policy_clf


@pytest.mark.parametrize("fitted_model", [0, 1, 2, 3, 4, 5], indirect=True)
def test_dump_model(fitted_model, get_dummy_data, request, tmpdir):
    new_data = get_dummy_data(seed=42)
    dump_model(tmpdir, fitted_model)
    assert os.path.exists(os.path.join(tmpdir, "final_model.dill"))
    with open(os.path.join(tmpdir, "final_model.dill"),
              "rb") as input_file_stream:
        fitted_model_reload = dill.load(input_file_stream)
    assert id(fitted_model) != id(fitted_model_reload)
    assert fitted_model.steps[0][1].vocabulary_ == fitted_model_reload.steps[
        0][1].vocabulary_
    assert (fitted_model.steps[0][1].idf_ == fitted_model_reload.steps[0]
            [1].idf_).all()
    assert (fitted_model.predict_proba(
        new_data.X) == fitted_model_reload.predict_proba(new_data.X)).all()
