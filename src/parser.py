#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import ArgparseFormatter, file_path
import argparse
import os


def get_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument(
        "--policies-csv",
        type=file_path,
        default=os.path.relpath(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data/1301_dataset.csv"
            )
        ),
        help="path to gold policies csv file",
    )
    parser.add_argument(
        "--scoring", type=str, default="roc_auc", help="scoring metric for GridSearchCV"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="global random seed for RNGs"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5, help="number of cross-validation splits"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="number of parallel jobs, specify -1 to use all processors",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=0.99,
        help="precision threshold to match",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level",
    )
    parser.add_argument("--debug", action="store_true", help="flag to debug script")
    return parser
