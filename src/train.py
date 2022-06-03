#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
from utils import (ArgparseFormatter, file_path, add_stream_handler, timestamp,
                   add_file_handler)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import argparse
import logging
import ipdb
import json
import dill
import os
import re

# define global dill setting
dill.settings["recurse"] = True

# define module's logger
LOGGER = logging.getLogger(__name__)


def preprocess(document: str) -> str:
    document = re.sub(r"http\S+", "", document)
    document = re.sub(r"http", "", document)
    document = re.sub(r"@\S+", "", document)
    document = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", document)
    document = re.sub(r"@", "at", document)
    document = re.sub(r"\n", " ", document)
    return document.lower()


def get_closest_value_index(vector: np.ndarray, value: float) -> int:
    return np.argmin(np.abs(vector - value)).item()


def main(args: argparse.Namespace) -> None:
    # create run_dir
    run_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs",
                           "run_" + timestamp())
    os.makedirs(run_dir, exist_ok=True)

    # update logger
    global LOGGER
    add_file_handler(LOGGER, args.logging_level,
                     os.path.join(run_dir, "session.log"))

    # log and dump args file
    args_file = os.path.join(run_dir, "args.json")
    LOGGER.info(args)
    LOGGER.info("Dumping arguments to disk: %s" % args_file)
    with open(args_file, "w") as output_file_stream:
        json.dump(vars(args), output_file_stream)

    # read in csv data and standardize
    df = pd.read_csv(args.policies_csv)

    # define inputs and outputs
    X = df["policy_text"].tolist()
    y = df["is_policy"].apply(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.random_seed)

    # define pipeline for classifier
    policy_clf = Pipeline([
        ("vect", TfidfVectorizer(stop_words="english",
                                 preprocessor=preprocess)),
        ("clf",
         RandomForestClassifier(class_weight="balanced",
                                random_state=args.random_seed)),
    ])
    LOGGER.info("Base classifier: %s" % policy_clf)

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
            "clf__min_samples_leaf": (2, 3)
        }
        LOGGER.info("Grid-search parameters %s" % parameters)

    # define grid-search model wrapper
    gs_policy_clf = GridSearchCV(policy_clf,
                                 parameters,
                                 scoring=args.scoring,
                                 cv=args.cv_splits,
                                 n_jobs=args.n_jobs,
                                 verbose=4)

    # fit on data
    gs_policy_clf.fit(X_train, y_train)
    LOGGER.info("Cross-validation with grid-search complete")
    LOGGER.info("Best model: %s" % gs_policy_clf.best_estimator_)

    # log and dump CV metrics
    cv_metrics_file = os.path.join(run_dir, "cv_metrics.json")
    LOGGER.info("Dumping CV metrics to disk: %s" % cv_metrics_file)
    with open(cv_metrics_file, "w") as output_file_stream:
        json.dump(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in gs_policy_clf.cv_results_.items()
            }, output_file_stream)

    # get test set probabilities
    y_probs_test = gs_policy_clf.best_estimator_.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_test, y_probs_test)

    # find the nearest precision to specified precision threshold
    index = get_closest_value_index(precision, args.precision_threshold)
    precision = precision[index]
    recall = recall[index]
    threshold = thresholds[index]
    LOGGER.info("Optimization condition satisfied")
    LOGGER.info("Precision: %s | Recall: %s | Threshold: %s" %
                (precision, recall, threshold))

    # log and dump PR metrics
    threshold_metrics_file = os.path.join(run_dir, "threshold_metrics.json")
    LOGGER.info("Dumping threshold metrics to disk: %s" %
                threshold_metrics_file)
    with open(threshold_metrics_file, "w") as output_file_stream:
        json.dump(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall
            }, output_file_stream)

    # refit final model
    LOGGER.info("Refitting final model on all data available")
    final_model = gs_policy_clf.best_estimator_.fit(X, y)
    final_model_file = os.path.join(run_dir, "final_model.dill")

    # dump model
    LOGGER.info("Dumping final model to disk: %s" % final_model_file)
    with open(final_model_file, "wb") as output_file_stream:
        dill.dump(final_model, output_file_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument("--policies-csv",
                        type=file_path,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "data/1301_dataset.csv"),
                        help="path to gold policies csv file")
    parser.add_argument("--scoring",
                        type=str,
                        default="roc_auc",
                        help="scoring metric for GridSearchCV")
    parser.add_argument("--random-seed",
                        type=int,
                        default=42,
                        help="global random seed for RNGs")
    parser.add_argument("--cv-splits",
                        type=int,
                        default=5,
                        help="number of cross-validation splits")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="number of parallel jobs, specify -1 to use all processors")
    parser.add_argument("--precision-threshold",
                        type=float,
                        default=0.99,
                        help="precision threshold to match")
    parser.add_argument(
        "--logging-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level")
    parser.add_argument("--debug",
                        action="store_true",
                        help="flag to debug script")
    LOGGER = logging.getLogger()
    add_stream_handler(LOGGER, parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
