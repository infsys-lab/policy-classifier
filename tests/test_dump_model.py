#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.train import dump_model, preprocess
import random
import pytest
import dill
import os

# define global dill setting
dill.settings["recurse"] = True

# define word possibilities
# source: https://stackoverflow.com/questions/29938804
NOUNS = ["puppy", "car", "rabbit", "girl", "monkey"]
VERBS = ["runs", "hits", "jumps", "drives", "barfs"]
ADV = ["crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally."]
ADJ = ["adorable", "clueless", "dirty", "odd", "stupid"]
SYMBOLS = ["@", "$", "."
           "%", "&"]
POSSIBILITIES = [NOUNS, SYMBOLS, VERBS, ADJ, ADV]


class RandomSequenceBinaryData:

    def __init__(self, dim=128, seed=0):
        random.seed(seed)
        self.X = [
            " ".join([random.choice(SUBSET) for SUBSET in POSSIBILITIES])
            for _ in range(dim)
        ]
        self.y = [random.randint(0, 1) for _ in range(dim)]


@pytest.fixture
def fitted_model(request):
    dummy_data = RandomSequenceBinaryData(seed=request.param)
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
def test_dump_model(fitted_model, request, tmpdir):
    new_data = RandomSequenceBinaryData(seed=42)
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
