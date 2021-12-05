import argparse
import os
import sys
import signal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .al import _start_active_learning
from .console import console, print_paper
from .text import (
    build_corpus,
    build_training_set,
    preprocess_abstract,
    query_papers_from_keywords,
)


def _create_parser():
    parser = argparse.ArgumentParser(description="ALreview server command line.")

    subparsers = parser.add_subparsers()

    # "select" subparser to run active learning and collect more labeled papers
    select_parser = subparsers.add_parser(
        "select",
        help="Command line to run active learning and collect more labeled papers.",
    )
    select_parser.add_argument(
        "-i", "--input", dest="input_file", type=str, required=True
    )
    select_parser.add_argument(
        "-s", "--save", dest="saved_file", type=str, default=None
    )
    select_parser.add_argument("-kw", "--keywords", type=str, default="")
    select_parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        required=False,
        default="selection.csv",
    )

    # "predict" subparser to run predictions and collect predicted labelled papers
    predict_parser = subparsers.add_parser(
        "predict",
        help="Command line to run predictions on a corpus of papers using a training set of labeled papers.",
    )
    predict_parser.add_argument(
        "-i", "--input", dest="input_file", type=str, required=True
    )
    predict_parser.add_argument(
        "-O", "--oracle", dest="oracle_file", type=str, required=True
    )
    predict_parser.add_argument(
        "-o", "--output", dest="output_file", type=str, default="prediction.csv"
    )
    predict_parser.add_argument("-t", "--threshold", type=float, default=0.1)

    return parser


def _run_select(input_file, saved_file, output_file, keywords):

    console.print("Starting ALreview [u]selection[/] session", style="underline red")

    input_file = os.path.abspath(input_file)

    console.print(f" * input file: '{input_file}'", style="info")
    console.print(f" * output file: '{output_file}'", style="info")

    if saved_file:
        saved_file = os.path.abspath(saved_file)
        df_selected = pd.read_csv(saved_file, index_col=0, dtype={"id": str})
        console.print("Loading saved DataFrame...", style="info")
        console.print(df_selected)

        keywords = keywords.split(",")
        console.print(f"Query papers from keywords: {keywords}", style="info")

        df_keywords = query_papers_from_keywords(keywords, input_file)
        df_keywords["selected"] = np.nan
        df_selected = pd.concat([df_selected, df_keywords], axis=0, ignore_index=True)
        df_selected = df_selected[
            ~df_selected.duplicated("title", keep="first")
        ].reset_index(drop=True)
    else:
        keywords = keywords.split(",")
        console.print(f"Query papers from keywords: {keywords}", style="info")
        df_selected = query_papers_from_keywords(keywords, input_file)
        df_selected["selected"] = np.nan

    # selection
    console.print("[b]Proposing papers based on keywords[/]", style="info")
    user_input = None
    for i in range(len(df_selected)):
        row = df_selected.iloc[i]

        if not (np.isnan(row["selected"])):
            continue

        print_paper(row)

        # ask the user if the paper should be selected
        user_input = None
        while not (user_input in ["y", "n", "quit"]):
            user_input = console.input(
                f"[red]Select the paper [yellow]({i+1}/{len(df_selected)})[/]?[/] (y/n/quit): "
            )

        if user_input == "quit":
            break

        df_selected.at[i, "selected"] = 1 if user_input == "y" else 0
        df_selected.to_csv(output_file)

    if user_input == "quit":
        exit()

    console.print("Pre-selection is over", style="info")

    # build the corpus and fit the preprocessor
    corpus = build_corpus(input_file)
    vectorizer = TfidfVectorizer(preprocessor=preprocess_abstract)
    vectorizer.fit(corpus)

    # get the training, query sets
    X_training, y_training, X, y, _ = build_training_set(
        df_selected, input_file, vectorizer
    )

    _start_active_learning(X_training, y_training, X, y, df_selected, output_file)


def _run_predict(input_file, oracle_file, output_file, threshold):

    console.print("Starting ALreview [u]prediction[/] session", style="underline red")

    input_file = os.path.abspath(input_file)
    oracle_file = os.path.abspath(oracle_file)

    console.print(f" * input file: '{input_file}'", style="info")
    console.print(f" * oracle file: '{oracle_file}'", style="info")
    console.print(f" * output file: '{output_file}'", style="info")

    # load oracle with labeled papers
    console.print("Loading oracle file...", style="info")
    df_selected = pd.read_csv(oracle_file, index_col=0, dtype={"id": str})

    # load corpus and prepare vectorizer
    corpus = build_corpus(input_file)
    vectorizer = TfidfVectorizer(preprocessor=preprocess_abstract)
    vectorizer.fit(corpus)

    # get the training, query sets
    X_training, y_training, X, _, df_X = build_training_set(
        df_selected, input_file, vectorizer
    )

    console.print("Running predictions...", style="info")
    model = RandomForestClassifier().fit(X_training, y_training)
    y_proba = model.predict_proba(X)

    df_X["selected"] = y_proba.argmax(axis=1)
    uncertainty = 1 - y_proba.max(axis=1)
    df_X = df_X[(uncertainty < threshold)]

    new_selection = df_X["selected"].sum()

    console.print(
        f"Finding [green]{new_selection}[/] new papers of interest and [red]{len(df_X)-new_selection}[/] not of interest",
        style="info",
    )

    console.print("Saving the predictions...", style="info")
    df_predict = pd.concat([df_selected, df_X], axis=0, ignore_index=True)
    df_predict.to_csv(output_file)


def main():
    parser = _create_parser()
    args = parser.parse_args()

    # handle ctrl + c
    def handler(signum, frame):
        console.print("\nGoodbye!", style="danger")
        exit(0)

    signal.signal(signal.SIGINT, handler)
    

    try:
        if sys.argv[1] == "select":
            _run_select(**vars(args))
        elif sys.argv[1] == "predict":
            _run_predict(**vars(args))
        else:
            raise ValueError(f"Argument '{sys.argv[1]}' not valid!")
    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="danger")


if __name__ == "__main__":
    main()
    
