import math
from pathlib import Path

from erllm.dbpedia.access_dbpedia import (
    Entity,
    get_entity_by_id,
    get_number_of_entries,
    get_random_matches,
    is_match,
    to_str,
)
from erllm.dataset.dbpedia.token_blocking import (
    clean_block_purging,
    clean_block_statistics,
    clean_comparisons,
    clean_token_blocking,
)
import itertools
import random
import pandas as pd
from py_stringmatching import Levenshtein


def sample_matches(
    n_desired_matches: int, include_keys: bool, max_lev_sim: float
) -> set[tuple[Entity, Entity]]:
    """
    Based on known matches generate a sample of matches whoose Levenshtein similarity does not exceed max_lev_sim.

    Parameters:
    - n_desired_matches (int): The desired number of matches to generate.
    - include_keys (bool): A flag indicating whether to include keys in the string representation of entities.
    - max_lev_sim (float): The maximum allowed Levenshtein similarity score for a match.

    Returns:
    set[tuple[Entity, Entity]]: A set of tuples representing matched entity pairs.
    """
    levenshtein = Levenshtein()
    matches = set()
    skip_count = 0
    while True:
        candidates = set(get_random_matches(2 * n_desired_matches))
        for c in candidates:
            e0 = get_entity_by_id(c[0], "dbpedia0")
            e1 = get_entity_by_id(c[1], "dbpedia1")
            e0_str = to_str(e0, include_keys)
            e1_str = to_str(e1, include_keys)
            if levenshtein.get_sim_score(e0_str, e1_str) <= max_lev_sim:
                matches.add((e0, e1))
            else:
                skip_count += 1
            if len(matches) == n_desired_matches:
                print("skipped matches", skip_count)
                return matches


def sample_dbpedia(
    N: int,
    match_ratio: float,
    include_keys: bool = False,
    max_lev_sim: float = 1,
    purge_factor: float = 1,
) -> list[tuple[bool, Entity, Entity]]:
    random.seed(42)
    pairs: list[tuple[bool, Entity, Entity]] = []
    # Sample N * match_ratio entries from dbpedia matches
    n_desired_matches = int(N * match_ratio)
    # Sample N * (1 - match_ratio) entries by token blocking on N random entries
    matches = sample_matches(n_desired_matches, include_keys, max_lev_sim)
    pairs = [(True, e0, e1) for (e0, e1) in matches]
    n_desired_non_matches = N - n_desired_matches
    N_db0, N_db1 = get_number_of_entries("dbpedia0"), get_number_of_entries("dbpedia1")
    # get sufficient random entries from 0 to N_db0 -1
    random_ids0 = set(
        random.sample(range(N_db0), int(4 * math.sqrt(n_desired_non_matches)))
    )
    random_ids1 = set(
        random.sample(range(N_db1), int(4 * math.sqrt(n_desired_non_matches)))
    )
    entities0 = set()
    entities1 = set()
    for id in random_ids0:
        entities0.add(get_entity_by_id(id, "dbpedia0"))
    for id in random_ids1:
        entities1.add(get_entity_by_id(id, "dbpedia1"))
    token_blocks = clean_token_blocking(entities0, entities1, include_keys)
    print(clean_block_statistics(token_blocks))
    token_blocks = clean_block_purging(token_blocks, purge_factor)
    print("After Block purging")
    print(clean_block_statistics(token_blocks))
    potential_non_matches = set(clean_comparisons(token_blocks))
    # takes long time since no index on dbpedia_matches
    non_matches = filter(lambda x: not is_match(x[0], x[1]), potential_non_matches)
    # draw N * (1-match_ratio) entries from non_matches
    non_matches = set(itertools.islice(non_matches, int(N * (1 - match_ratio))))
    pairs.extend(
        [
            (
                False,
                get_entity_by_id(x[0], "dbpedia0"),
                get_entity_by_id(x[1], "dbpedia1"),
            )
            for x in non_matches
        ]
    )
    # Combine matches and non_matches to get the final set
    return pairs


def to_benchmark_csv(
    path: Path, pairs: list[tuple[bool, Entity, Entity]], include_keys=False
):
    # Extract information from pairs
    data = []
    columns = [
        "label",
        "table1.id",
        "table2.id",
        "table1.description",
        "table2.description",
    ]
    for is_match, entity1, entity2 in pairs:
        data.append(
            (
                int(is_match),
                entity1.id,
                entity2.id,
                to_str(entity1, include_keys),
                to_str(entity2, include_keys),
            )
        )

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=columns)
    # Write the DataFrame to a CSV file
    df.to_csv(path, index=True, index_label="_id")


CONFIGURATIONS = {
    "dbpedia": {
        "folder": Path("data/benchmark_datasets/existingDatasets/dbpedia10k"),
        "args": {"purge_factor": 1, "max_lev_sim": 1},
    },
    # create harder test dataset by purging large blocks and disallowing exact matches
    "dbpedia_harder": {
        "folder": Path("data/benchmark_datasets/existingDatasets/dbpedia10k_harder"),
        "args": {"purge_factor": 0.1, "max_lev_sim": 0.9},
    },
}
if __name__ == "__main__":
    cfg = CONFIGURATIONS["dbpedia_harder"]
    pairs = sample_dbpedia(10000, 0.05, include_keys=True, **cfg["args"])
    print(len(pairs))
    dbpedia_folder = cfg["folder"]
    dbpedia_folder.mkdir(parents=True, exist_ok=True)
    to_benchmark_csv(dbpedia_folder / "train.csv", pairs, include_keys=True)
