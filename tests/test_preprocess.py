#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import preprocess
import pytest


@pytest.mark.parametrize(
    "input_document, output_document",
    [("", ""), ("my website is https://example.com", "my website is "),
     ("my website is http://example.com", "my website is "),
     ("my email is person@example.com", "my email is person"),
     (".%&$", "    "),
     ("my email is person @ example dot com",
      "my email is person at example dot com"),
     ("what a\nnice day", "what a nice day"),
     ("WHAT A NICE DAY", "what a nice day")])
def test_preprocess(input_document, output_document):
    assert preprocess(input_document) == output_document
