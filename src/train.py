#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import dill
import ipdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from parser import get_train_parser
from utils import add_file_handler, add_stream_handler, timestamp

# define global dill setting
dill.settings["recurse"] = True

# define module's logger
LOGGER = logging.getLogger(__name__)


def get_run_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "runs", "run_" + timestamp()
    )


def dump_args(args: argparse.Namespace, run_dir: str) -> None:
    args_file = os.path.join(run_dir, "args.json")
    LOGGER.info(args)
    LOGGER.info("Dumping arguments to disk: %s" % args_file)
    with open(args_file, "w") as output_file_stream:
        json.dump(vars(args), output_file_stream)


def get_raw_data(args: argparse.Namespace) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(args.policies_csv)
    return df["policy_text"].tolist(), df["is_policy"].apply(int).tolist()


def preprocess(document: str) -> str:
    document = re.sub(r"http\S+", "", document)
    document = re.sub(r"http", "", document)
    document = re.sub(r"@\S+", "", document)
    document = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", document)
    document = re.sub(r"@", "at", document)
    document = re.sub(r"\n", " ", document)
    return document.lower()


def get_grid_search_hyperparameters(args: argparse.Namespace) -> Dict[Any, Any]:
    if args.debug:
        parameters = {"vect__ngram_range": [(1, 2)]}
        args.cv_splits = 2
        ipdb.set_trace()
    else:
        # define grid-search parameters
        parameters: Dict[Any, Any] = {  # type: ignore
            "vect__ngram_range": [(1, 2), (1, 4)],
            "vect__min_df": (0.1, 0.2, 0.3),
            "clf__max_depth": (10, 15),
            "clf__n_estimators": (100, 200),
            "clf__min_samples_leaf": (2, 3),
        }
        LOGGER.info("Grid-search parameters %s" % parameters)
    return parameters


def get_closest_value_index(vector: np.ndarray, value: float) -> int:
    return np.argmin(np.abs(vector - value)).item()


def dump_metrics(run_dir: str, metrics: Dict) -> None:
    metrics_file = os.path.join(run_dir, "metrics.json")
    LOGGER.info("Dumping metrics to disk: %s" % metrics_file)
    with open(metrics_file, "w") as output_file_stream:
        json.dump(metrics, output_file_stream)


def dump_model(run_dir: str, final_model: Pipeline) -> None:
    final_model_file = os.path.join(run_dir, "final_model.dill")
    LOGGER.info("Dumping final model to disk: %s" % final_model_file)
    with open(final_model_file, "wb") as output_file_stream:
        dill.dump(final_model, output_file_stream)


def main(args: argparse.Namespace) -> None:
    # create run_dir
    run_dir = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # update logger
    global LOGGER
    add_file_handler(LOGGER, args.logging_level, os.path.join(run_dir, "session.log"))

    # log and dump args file
    dump_args(args, run_dir)

    # parse raw data
    X, y = get_raw_data(args)

    # split raw data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.random_seed
    )

    # initialize policy classifier (pipeline)
    policy_clf = Pipeline(
        [
            ("vect", TfidfVectorizer(stop_words="english", preprocessor=preprocess)),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=args.random_seed
                ),
            ),
        ]
    )
    LOGGER.info("Base classifier: %s" % policy_clf)

    # retrieve grid-search parameter space
    parameters = get_grid_search_hyperparameters(args)

    # define grid-search model wrapper
    gs_policy_clf = GridSearchCV(
        policy_clf,
        parameters,
        scoring=args.scoring,
        cv=args.cv_splits,
        n_jobs=args.n_jobs,
        verbose=4,
    )

    # fit on data and log current state
    gs_policy_clf.fit(X_train, y_train)
    LOGGER.info("Cross-validation with grid-search complete")
    LOGGER.info("Best model: %s" % gs_policy_clf.best_estimator_)

    # get test set probabilities
    y_probs_test = gs_policy_clf.best_estimator_.predict_proba(X_test)[:, 1]

    # compute precision-recall data based on probabilities
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs_test)

    clf_report = classification_report(
        y_test, (y_probs_test >= 0.5).astype(int), output_dict=True
    )

    # find the nearest precision to specified precision threshold
    # NOTE: last elements of precisiona and recall are constants for graphs
    # NOTE: we do not need to consider them and can exclude them as below
    index = get_closest_value_index(precision[:-1], args.precision_threshold)

    # get precision, recall and threshold that are closest to desired
    precision, recall, threshold = precision[index], recall[index], thresholds[index]
    LOGGER.info("Optimization condition satisfied")
    LOGGER.info(
        "Precision: %s | Recall: %s | Threshold: %s" % (precision, recall, threshold)
    )

    # compile all metrics together here
    metrics = {
        "cv_metrics": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in gs_policy_clf.cv_results_.items()
        },
        "clf_report": clf_report,
        "threshold_metrics": {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
        },
    }

    # log and dump all metrics
    dump_metrics(run_dir, metrics)

    # refit final model
    LOGGER.info("Refitting final model on all data available")
    final_model = gs_policy_clf.best_estimator_.fit(X, y)

    # dump model
    dump_model(run_dir, final_model)


if __name__ == "__main__":
    parser = get_train_parser()
    LOGGER = logging.getLogger()
    add_stream_handler(LOGGER, parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
