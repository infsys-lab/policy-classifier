#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


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


def main() -> None:
    # read in csv data and standardize
    df = pd.read_csv("data/1301_dataset.csv")
    df = standardize_text(df, "policy_text")

    # perform sanity check to ensure no html tags
    assert not df["policy_text"].str.contains(r"<|>", regex=True).any()

    # define inputs and outputs
    X = df["policy_text"]
    y = df["is_policy"].apply(int)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=42)

    # define pipeline for classifier
    policy_clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words="english")),
        ('clf', RandomForestClassifier(class_weight="balanced",
                                       random_state=42)),
    ])

    # define grid-search parameters
    parameters = {
        "vect__ngram_range": [(1, 2), (1, 4)],
        "vect__min_df": (0.1, 0.2),
        "clf__max_depth": (10, 15),
        "clf__n_estimators": (100, 200),
        "clf__min_samples_leaf": (2, 3)
    }

    # define grid-search model wrapper
    gs_policy_clf = GridSearchCV(policy_clf,
                                 parameters,
                                 scoring="roc_auc",
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=4)

    # fit on data
    gs_policy_clf.fit(X_train, y_train)

    # get test set probabilities
    y_probs_test = gs_policy_clf.best_estimator_.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs_test)

    # find the nearest precision to 0.99
    index = np.argmin(np.abs(precision - 0.99))
    threshold = thresholds[index]

    # re-train final model
    final_model = gs_policy_clf.best_estimator_.fit(X, y)

    import ipdb
    ipdb.set_trace()
    pass


# TODO: try auto-sklearn for possibly better performance
# TODO: add argument parser -> csv file, random seed, number of splits, threshold
# TODO: use directory structure to write logs and model
# TODO: add necessary input to main
# TODO: catch errors here in case does threshold does not exist
# TODO: log the result of threshold analysis
# TODO: double-check reproducibility
# TODO: dump all relevant files at the end

if __name__ == "__main__":
    main()
