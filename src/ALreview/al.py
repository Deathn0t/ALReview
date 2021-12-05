import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack

from .console import console


def _start_active_learning(X_training, y_training, X, y, df_selected, output_file):

    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling,
        X_training=X_training,
        y_training=y_training,
    )

    # active learning
    idx = 0
    size_pool = len(df_selected) + len(y.df)
    while idx < len(y.df):

        perc_explored = len(df_selected) / size_pool * 100
        perc_selected = df_selected["selected"].sum() / len(df_selected) * 100

        console.print(
            f"\n [bold magenta]->[/] [italic yellow]Percentage explored: [/] {perc_explored:.4f} % - [italic yellow] Percentage selected: [/] {perc_selected:.4f} % ({int(df_selected['selected'].sum())}/{len(df_selected)})"
        )

        query_idx, query_instance = learner.query(X)
        X_query = X[query_idx].reshape(1, -1)
        y_pred = learner.predict(X_query)[0]
        console.print(
            f" [bold magenta]->[/] [italic yellow]ALreview predicted:[/] {'[green]YES[/]' if y_pred else '[red]NO[/]'}"
        )
        y_query = y[query_idx].reshape(-1)

        paper_query = y.get_paper(query_idx)
        paper_query["selected"] = y_query[0]
        df_selected = df_selected.append(paper_query, ignore_index=True)
        df_selected.to_csv(output_file)

        X = vstack([X[: query_idx[0]], X[query_idx[0] + 1 :]]).toarray()
        y.drop(query_idx)

        learner.teach(X_query, y_query)
