from access_dbpedia import (
    Entity,
    get_entity_by_id,
    get_number_of_entries,
    get_random_matches,
    is_match,
)
from token_blocking import (
    clean_comparisons,
    clean_token_blocking,
    clean_block_statistics,
)
import itertools
import random


def sample_dbpedia(N, match_ratio) -> list[tuple[bool, Entity, Entity]]:
    pairs: list[tuple[bool, Entity, Entity]] = []
    # Sample N * match_ratio entries from dbpedia matches
    n_desired_matches = int(N * match_ratio)
    n_desired_non_matches = N - n_desired_matches

    N_db0, N_db1 = get_number_of_entries("dbpedia0"), get_number_of_entries("dbpedia1")
    # get  n_desired_non_matches random entries from 0 to N_db0 -1
    random_ids0 = set(random.sample(range(N_db0), n_desired_non_matches))
    random_ids1 = set(random.sample(range(N_db1), n_desired_non_matches))
    entities0 = set()
    entities1 = set()
    for id in random_ids0:
        entities0.add(get_entity_by_id(id, "dbpedia0"))
    for id in random_ids1:
        entities1.add(get_entity_by_id(id, "dbpedia1"))
    token_blocks = clean_token_blocking(entities0, entities1)
    potential_non_matches = set(clean_comparisons(token_blocks))
    # takes long time since no index on dbpedia_matches
    non_matches = filter(lambda x: not is_match(x[0], x[1]), potential_non_matches)
    clean_block_statistics(token_blocks)
    # Sample N * (1 - match_ratio) entries by token blocking on N random entries
    matches = set(get_random_matches(n_desired_matches))
    entities0 = set()
    entities1 = set()

    for m in matches:
        entities0.add(get_entity_by_id(m[0], "dbpedia0"))
        entities1.add(get_entity_by_id(m[1], "dbpedia1"))

    pairs = [
        (True, get_entity_by_id(x[0], "dbpedia0"), get_entity_by_id(x[1], "dbpedia1"))
        for x in matches
    ]
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


if __name__ == "__main__":
    print(sample_dbpedia(2000, 0.05))
