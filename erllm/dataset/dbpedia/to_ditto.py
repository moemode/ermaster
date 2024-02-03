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


def entities_from_dbpedia_csv(
    dbpedia_csv: Path,
) -> Iterable[tuple[Entity, Entity, int]]:
    """
    Extracts entities from a DBpedia CSV file.

    Args:
        dbpedia_csv (Path): The path to the DBpedia CSV file.

    Returns:
        Iterable[tuple[Entity, Entity, int]]: A list of tuples containing the entities and their labels.
    """
    df = pd.read_csv(dbpedia_csv)
    pair_ids = [
        (row["table1.id"], row["table2.id"], row["label"]) for _, row in df.iterrows()
    ]
    pair_entities = []
    for id0, id1, label in pair_ids:
        pair_entities.append(
            (
                get_entity_by_id(id0, "dbpedia0"),
                get_entity_by_id(id1, "dbpedia1"),
                label,
            )
        )
    return pair_entities


def dbpedia_to_test_train_ditto(
    dbpedia_test_csv: Path, dbpedia_reserve_csv: Path, label_fraction: float
):
    """
    Extracts entities from a DBpedia CSV file.

    Args:
        dbpedia_csv (Path): The path to the DBpedia CSV file.

    Returns:
        Iterable[tuple[Entity, Entity, int]]: A list of tuples containing the entities and their labels.
    """
    seed = 123
    df_test_train = pd.read_csv(dbpedia_test_csv)
    df_valid = pd.read_csv(dbpedia_reserve_csv)
    df_test_train_ids = [
        (row["table1.id"], row["table2.id"]) for _, row in df_test_train.iterrows()
    ]
    # remove all rows with ids in df_test_train_ids from df_valid
    df_valid = df_valid[
        ~df_valid[["table1.id", "table2.id"]]
        .apply(tuple, axis=1)
        .isin(df_test_train_ids)
    ]
    N, N_pos, N_neg = (
        len(df_test_train),
        sum(df_test_train["label"]),
        len(df_test_train) - sum(df_test_train["label"]),
    )
    N_train, N_train_pos = (label_fraction * N, round(label_fraction * N_pos))
    N_train_neg = N_train - N_train_pos

    # sample N_train_pos rows with label 1 and N_train_neg rows with label 0
    train_ids = df_test_train[df_test_train["label"] == 1].sample(N_train_pos, seed)
    train_ids = train_ids.append(
        df_test_train[df_test_train["label"] == 0].sample(N_train_neg, seed)
    )

    test_entities = []
    for id0, id1, label in test_ids:
        test_entities.append(
            (
                get_entity_by_id(id0, "dbpedia0"),
                get_entity_by_id(id1, "dbpedia1"),
                label,
            )
        )
    return test_entities


if __name__ == "__main__":
    label_fraction = 0.15
    dbpedia1250_csv = (
        DATA_FOLDER_PATH
        / "benchmark_datasets/existingDatasets/dbpedia10k_1250/test.csv"
    )
    dbpedia10k_csv = (
        DATA_FOLDER_PATH / "benchmark_datasets/existingDatasets/dbpedia10k/train.csv"
    )
    ditto_file = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/ditto/dbpedia10k_1250/test.txt"
    )
    dbpedia_to_test_train_ditto(dbpedia1250_csv, dbpedia10k_csv, label_fraction)
    """
    labeled_pairs = list(entities_from_dbpedia_csv(dbpedia1250_csv))
    ditto_file.parent.mkdir(parents=True, exist_ok=True)
    to_ditto(labeled_pairs, ditto_file)
    """
