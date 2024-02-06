from pathlib import Path
from typing import Iterable
import pandas as pd
from erllm import DATA_FOLDER_PATH
from erllm.dataset.entity import Entity, OrderedEntity
from typing import Iterable, Tuple
from pathlib import Path

LabeledPairs = Iterable[
    Tuple[int | bool, Entity | OrderedEntity, Entity | OrderedEntity]
]


def to_ditto(
    labeled_pairs: LabeledPairs,
    ditto_file: Path,
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
        for label, entity0, entity1 in labeled_pairs:
            line = f"{entity0.to_ditto_str()}\t{entity1.to_ditto_str()}\t{int(label)}\n"
            file.write(line)


def to_ditto_task(
    train: LabeledPairs, valid: LabeledPairs, test: LabeledPairs, task_folder: Path
) -> None:
    task_folder.mkdir(parents=True, exist_ok=True)
    for labeled_pairs, stem in zip((train, valid, test), ("train", "valid", "test")):
        ditto_file = task_folder / f"{stem}.txt"
        to_ditto(labeled_pairs, ditto_file)


def sample_df(df: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    # sample N_train_pos rows with label 1 and N_train_neg rows with label 0
    pos = df[df["label"] == 1].sample(n_pos, random_state=seed)
    # Sample N_train_neg rows with label 0
    neg = df[df["label"] == 0].sample(n_neg, random_state=seed)
    # Concatenate the positive and negative samples to create the final training set
    return pd.concat([pos, neg])


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
