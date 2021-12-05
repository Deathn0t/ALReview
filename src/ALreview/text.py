import json
import re
import string

import pandas as pd
import numpy as np

from .console import console, print_paper


def decode_line(line):
    paper = json.loads(line)
    return paper


def preprocess_title(text):
    text = text.lower().replace("-", " ").replace("\n", "").replace("  ", " ").strip()
    return text


def preprocess_abstract(text):
    text = text.lower().replace("-", " ").replace("\n", " ").strip()
    text = re.sub(r"[{}]".format(string.punctuation), " ", text)
    return text


def line_iterator(file):
    """
    Access the metadata of a specific category one by one.
    """
    with open(file, "r") as f:
        for line in f:
            yield line


def paper_iterator(file):
    for line in line_iterator(file):
        paper = decode_line(line)

        if (
            "copyright infringement" in paper["abstract"]
            or "this article was withdrawn" in paper["abstract"]
        ):
            continue

        yield paper

def paper_to_dict(paper):
    paper = {
                "id": str(paper["id"]),
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper["authors_parsed"],
                "update_date": paper["update_date"],
            }
    return paper

def query_papers_from_keywords(kws, file):
    collected_papers = []

    metadata = paper_iterator(file)
    for ind, paper in enumerate(metadata):

        if any(kw in preprocess_title(paper["title"]) for kw in kws) or any(
            kw in preprocess_abstract(paper["abstract"]) for kw in kws
        ):
            paper = paper_to_dict(paper)
            collected_papers.append(paper)

    df = pd.DataFrame(data=collected_papers)

    console.print(f"Found {len(df)} papers with keywords.", style="info")

    return df

def dataframe_from_file(file):
    collected_papers = []

    metadata = paper_iterator(file)
    for ind, paper in enumerate(metadata):

        paper = paper_to_dict(paper)
        collected_papers.append(paper)

    df = pd.DataFrame(data=collected_papers)

    return df


def build_corpus(file):

    # create the corpus of abstract
    corpus = []
    metadata = paper_iterator(file)
    for ind, paper in enumerate(metadata):

        corpus.append(paper["abstract"])

    return corpus


class TextQuery:
    def __init__(self, df) -> None:
        self.df = df

    def __getitem__(self, item):
        row = self.df.iloc[item].to_dict("records")[0]

        print_paper(row)

        user_input = None
        while not (user_input in ["y", "n", "quit"]):
            user_input = console.input("[red]Select the paper?[/] (y/n/quit): ")

        if user_input == "quit":
            raise KeyboardInterrupt
            
        val = 1 if user_input == "y" else 0
        return np.array([val])

    def get_paper(self, item):
        return self.df.iloc[item].to_dict("records")[0]

    def drop(self, idx):
        self.df = self.df.drop(idx, axis=0)


def build_training_set(df_selected, file, vectorizer=None):

    # collect possible candidates (i.e., papers that are not selected yet)
    collected_papers = []
    metadata = paper_iterator(file)
    for ind, paper in enumerate(metadata):

        if not (paper["id"] in df_selected["id"].values):

            paper = paper_to_dict(paper)
            collected_papers.append(paper)

    df_pool = pd.DataFrame(data=collected_papers)

    X_training = df_selected["abstract"].values
    y_training = df_selected["selected"].values.reshape(-1)

    X = df_pool["abstract"].values
    y = TextQuery(df_pool)

    if vectorizer:
        X_training = vectorizer.transform(X_training)
        X = vectorizer.transform(X)

    return X_training, y_training, X, y, df_pool
