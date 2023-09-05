#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from argparse import Namespace

import pytest

from train import dump_args


@pytest.mark.parametrize(
    "args",
    [
        Namespace(a=1, b=2),
        Namespace(a=1.0, b=2.0),
        Namespace(a=True, b=True),
        Namespace(a="test", b="test"),
        Namespace(a=None, b=None),
        Namespace(a=1, b=2.0, c=True, d="test", e=None),
    ],
)
def test_dump_args(args, tmpdir):
    dump_args(args, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "args.json"))
    with open(os.path.join(tmpdir, "args.json")) as input_file_stream:
        args_reload = json.load(input_file_stream)
    assert vars(args) == args_reload
