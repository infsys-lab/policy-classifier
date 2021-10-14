#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from joblib import dump
from utils import (  # type: ignore
    ArgparseFormatter, file_path, get_formatted_logger, timestamp,
    add_file_handler)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import argparse
import json
import os


def standardize_text(df: pd.DataFrame, text_field: str) -> pd.DataFrame:
    df[text_field] = df[text_field].str.replace(r"http\S+", "", regex=True)
    df[text_field] = df[text_field].str.replace(r"http", "", regex=True)
    df[text_field] = df[text_field].str.replace(r"@\S+", "", regex=True)
    df[text_field] = df[text_field].str.replace(
        r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", regex=True)
    df[text_field] = df[text_field].str.replace(r"@", "at", regex=True)
    df[text_field] = df[text_field].str.replace(r"\n", " ", regex=True)
    df[text_field] = df[text_field].str.lower()
    return df


def main(args: argparse.Namespace) -> None:
    # create run_dir
    run_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs",
                           "run_" + timestamp())
    os.makedirs(run_dir, exist_ok=True)

    # update logger
    global LOGGER
    add_file_handler(LOGGER, os.path.join(run_dir, "session.log"))

    # log and dump args file
    args_file = os.path.join(run_dir, "args.json")
    LOGGER.info(args)
    LOGGER.info("Dumping arguments to disk: %s" % args_file)
    with open(args_file, "w") as output_file_stream:
        json.dump(vars(args), output_file_stream)

    # read in csv data and standardize
    df = pd.read_csv(args.policies_csv)
    df = standardize_text(df, "policy_text")

    # perform sanity check to ensure no html tags
    assert not df["policy_text"].str.contains(r"<|>", regex=True).any()

    # define inputs and outputs
    X = df["policy_text"]
    y = df["is_policy"].apply(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.random_seed)

    # define pipeline for classifier
    policy_clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words="english")),
        ('clf',
         RandomForestClassifier(class_weight="balanced",
                                random_state=args.random_seed)),
    ])
    LOGGER.info("Base classifier: %s" % policy_clf)

    # define grid-search parameters
    parameters = {
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
    index = np.argmin(np.abs(precision - args.precision_threshold))
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
    final_policy_clf = gs_policy_clf.best_estimator_.fit(X, y)
    dump(final_policy_clf, os.path.join(run_dir, "final_model.joblib"))


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
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="set logging level")
    LOGGER = get_formatted_logger(parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
