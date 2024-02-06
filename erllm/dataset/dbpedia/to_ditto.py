from pathlib import Path
import pandas as pd
from typing import Tuple
from pathlib import Path


def sample_df(df: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    # sample N_train_pos rows with label 1 and N_train_neg rows with label 0
    pos = df[df["label"] == 1].sample(n_pos, random_state=seed)
    # Sample N_train_neg rows with label 0
    neg = df[df["label"] == 0].sample(n_neg, random_state=seed)
    # Concatenate the positive and negative samples to create the final training set
    return pd.concat([pos, neg])


def dbpedia_to_train_valid_test(
    dbpedia_test_csv: Path, dbpedia_reserve_csv: Path, label_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts entities from a DBpedia CSV file.

    Args:
        dbpedia_csv (Path): The path to the DBpedia CSV file.

    Returns:
        Iterable[tuple[Entity, Entity, int]]: A list of tuples containing the entities and their labels.
    """
    seed = 123
    df_test = pd.read_csv(dbpedia_test_csv)
    df_reserve = pd.read_csv(dbpedia_reserve_csv)
    # remove all rows with ids in df_test_train_ids from df_valid
    df_test_train_ids = [
        (row["table1.id"], row["table2.id"]) for _, row in df_test.iterrows()
    ]
    df_reserve = df_reserve[
        ~df_reserve[["table1.id", "table2.id"]]
        .apply(tuple, axis=1)
        .isin(df_test_train_ids)
    ]
    N, N_pos, N_neg = (
        len(df_test),
        sum(df_test["label"]),
        len(df_test) - sum(df_test["label"]),
    )
    N_train, N_train_pos = (
        int(round(label_fraction * N)),
        int(round(label_fraction * N_pos)),
    )
    N_train_neg = N_train - N_train_pos
    df_train = sample_df(df_test, N_train_pos, N_train_neg, seed)
    df_valid = sample_df(df_reserve, N_train_pos, N_train_neg, seed)
    return df_train, df_valid, df_test
