from pathlib import Path
from typing import Iterable
import pandas as pd
from typing import Iterable, Tuple, Set
from pathlib import Path
import random
from erllm.dataset.entity import Entity, OrderedEntity

LabeledPairs = Iterable[
    Tuple[int | bool, Entity | OrderedEntity, Entity | OrderedEntity]
]

LabeledPairsSet = Set[Tuple[int | bool, Entity | OrderedEntity, Entity | OrderedEntity]]


def to_ditto_task(
    train: LabeledPairs, valid: LabeledPairs, test: LabeledPairs, task_folder: Path
) -> None:
    task_folder.mkdir(parents=True, exist_ok=True)
    for labeled_pairs, stem in zip((train, valid, test), ("train", "valid", "test")):
        ditto_file = task_folder / f"{stem}.txt"
        pairs_to_ditto(labeled_pairs, ditto_file)


def pairs_to_ditto(
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


def ditto_split(
    labeled_pairs: LabeledPairs,
    train_fraction: float,
    valid_fraction: float,
    seed: int,
) -> Tuple[LabeledPairs, LabeledPairs, LabeledPairs]:
    labeled_pairs = set(labeled_pairs)
    N = len(labeled_pairs)
    pos_ratio = sum(label for label, _, _ in labeled_pairs) / len(labeled_pairs)
    N_train_valid = int(round(len(labeled_pairs) * (train_fraction + valid_fraction)))
    N_train = int(round(len(labeled_pairs) * train_fraction))
    N_valid = N_train_valid - N_train
    train = sample(labeled_pairs, pos_ratio, N_train, seed)
    labeled_pairs.difference_update(train)
    valid = sample(labeled_pairs, pos_ratio, N_valid, seed)
    labeled_pairs.difference_update(valid)
    assert len(train) + len(valid) + len(labeled_pairs) == N
    assert train.intersection(valid) == set()
    assert train.intersection(labeled_pairs) == set()
    assert valid.intersection(labeled_pairs) == set()
    return train, valid, labeled_pairs


def sample(
    labeled_pairs: LabeledPairsSet, pos_ratio: float, N: int, seed: int
) -> LabeledPairsSet:
    random.seed(seed)
    N_pos = int(round(N * pos_ratio))
    N_neg = N - N_pos
    # sample N_train_pos rows with label 1 and N_train_neg rows with label 0
    pos = [pair for pair in labeled_pairs if pair[0] == 1]
    neg = [pair for pair in labeled_pairs if pair[0] == 0]
    pos_sample = set(
        random.sample(
            pos,
            N_pos,
        )
    )
    neg_sample = set(random.sample(neg, N_neg))
    return pos_sample.union(neg_sample)


def sample_df(df: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    # sample N_train_pos rows with label 1 and N_train_neg rows with label 0
    pos = df[df["label"] == 1].sample(n_pos, random_state=seed)
    # Sample N_train_neg rows with label 0
    neg = df[df["label"] == 0].sample(n_neg, random_state=seed)
    # Concatenate the positive and negative samples to create the final training set
    return pd.concat([pos, neg])
