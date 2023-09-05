#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import random
import pytest


class RandomSequenceBinaryData:
    # define word possibilities
    # source: https://stackoverflow.com/questions/29938804
    NOUNS = ["puppy", "car", "rabbit", "girl", "monkey"]
    VERBS = ["runs", "hits", "jumps", "drives", "barfs"]
    ADV = ["crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally."]
    ADJ = ["adorable", "clueless", "dirty", "odd", "stupid"]
    SYMBOLS = ["@", "$", ".", "%", "&"]
    POSSIBILITIES = [NOUNS, SYMBOLS, VERBS, ADJ, ADV]

    def __init__(self, dim=128, seed=0):
        random.seed(seed)
        self.seed = seed
        self.X = [
            " ".join([random.choice(self.SUBSET) for self.SUBSET in self.POSSIBILITIES])
            for _ in range(dim)
        ]
        self.y = [random.randint(0, 1) for _ in range(dim)]


@pytest.fixture
def get_dummy_data():
    return RandomSequenceBinaryData


def pytest_configure():
    logging.disable(logging.CRITICAL)
