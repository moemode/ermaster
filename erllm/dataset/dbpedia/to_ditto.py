from pathlib import Path
from typing import Iterable
import pandas as pd
from erllm import DATA_FOLDER_PATH
from erllm.dataset.dbpedia.access_dbpedia import get_entity_by_id
from erllm.dataset.entity import Entity


from typing import Iterable, Tuple
from pathlib import Path


def to_ditto(
    labeled_pairs: Iterable[Tuple[Entity, Entity, int]], ditto_file: Path
) -> None:
    """
    Convert labeled pairs of entities to a Ditto file.

    Args:
        labeled_pairs (Iterable[Tuple[Entity, Entity, int]]): A collection of labeled entity pairs.
        ditto_file (Path): The path to the Ditto file to be created.
    Returns:
        None
    """
    with open(ditto_file, "w", encoding="utf-8") as file:
        for entity0, entity1, label in labeled_pairs:
            line = f"{entity0.to_ditto_str()}\t{entity1.to_ditto_str()}\t{label}\n"
            file.write(line)


def entities_from_dbpedia_df(
    df: pd.DataFrame,
) -> Iterable[tuple[int, Entity, Entity]]:
    """
    Extracts entities from a DBpedia CSV file.

    Args:
        dbpedia_csv (Path): The path to the DBpedia CSV file.

    Returns:
        Iterable[tuple[Entity, Entity, int]]: A list of tuples containing the entities and their labels.
    """
    pair_ids = [
        (row["table1.id"], row["table2.id"], row["label"]) for _, row in df.iterrows()
    ]
    pair_entities = []
    for id0, id1, label in pair_ids:
        pair_entities.append(
            (
                int(label),
                get_entity_by_id(id0, "dbpedia0"),
                get_entity_by_id(id1, "dbpedia1"),
            )
        )
    return pair_entities


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


if __name__ == "__main__":
    label_fraction = 0.15
    dbpedia1250_csv = (
        DATA_FOLDER_PATH
        / "benchmark_datasets/existingDatasets/dbpedia10k_1250/test.csv"
    )
    dbpedia10k_csv = (
        DATA_FOLDER_PATH / "benchmark_datasets/existingDatasets/dbpedia10k/train.csv"
    )
    dbpedia_ditto_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/ditto/dbpedia"
    )
    dbpedia_ditto_folder.mkdir(parents=True, exist_ok=True)
    dfs = dbpedia_to_train_valid_test(dbpedia1250_csv, dbpedia10k_csv, label_fraction)
    for df, stem in zip(dfs, ("train", "valid", "test")):
        entities = entities_from_dbpedia_df(df)
        ditto_file = dbpedia_ditto_folder / f"{stem}.txt"
        to_ditto(entities, ditto_file)

    """
    df = pd.read_csv(dbpedia1250_csv)
    labeled_pairs = list(entities_from_dbpedia_csv(df))
    ditto_file.parent.mkdir(parents=True, exist_ok=True)
    to_ditto(labeled_pairs, ditto_file)
    """
