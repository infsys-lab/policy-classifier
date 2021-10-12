#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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


# TODO: add argument parser
# TODO: add necessary input to main

main()
