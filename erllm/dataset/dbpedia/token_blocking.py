"""
Provides functions for token blocking and clean token blocking in entity resolution tasks.
"""

# Rest of the code...
from typing import Dict, Set, Iterable
from erllm.dataset.dbpedia.access_dbpedia import (
    Entity,
    get_entity_by_id,
    get_random_matches,
)
from erllm.dataset.entity import tokens
import itertools
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd


def token_blocking(entities: Set[Entity]) -> Dict[str, Set[int]]:
    """
    Perform token blocking on entities for Dirty ER.

    Args:
        entities (Set[Entity]): A set of Entity objects.

    Returns:
        Dict[str, Set[int]]: A dictionary mapping tokens to sets of entity IDs.
    """
    blocks: Dict[str, Set[int]] = {}
    for e in entities:
        for t in tokens(e):
            if t not in blocks:
                blocks[t] = set()
            blocks[t].add(e.id)
    for k in list(blocks.keys()):
        if len(blocks[k]) < 2:
            del blocks[k]
    return blocks


def comparisons(blocks: Dict[str, Set[int]]) -> Iterable[tuple[int, int]]:
    """
    Generate comparisons based on blocks produced by token_blocking.

    Args:
        blocks (Dict[str, Set[int]]): A dictionary mapping tokens to sets of entity IDs.

    Returns:
        Iterable[tuple[int, int]]: An iterable of entity ID pairs.
    """
    for b in blocks.values():
        for e0, e1 in itertools.combinations(b, 2):
            yield e0, e1


def block_statistics(blocks: Dict[str, Set[int]]):
    """
    Print statistics about token blocks.

    Args:
        blocks (Dict[str, Set[int]]): A dictionary mapping tokens to sets of entity IDs.
    """
    print(f"Number of blocks: {len(blocks)}")
    print(f"Average block size: {sum(map(len, blocks.values())) / len(blocks)}")
    print(f"Maximum block size: {max(map(len, blocks.values()))}")
    print(f"Minimum block size: {min(map(len, blocks.values()))}")


def clean_token_blocking(
    entities0: Set[Entity], entities1: Set[Entity], include_keys=False
) -> Dict[str, tuple[Set[int], Set[int]]]:
    """
    Perform tocken blocking on two datasources (sets of entities) for Clean-Clean ER.

    Args:
        entities0 (Set[Entity]): A set of Entity objects from dataset 0.
        entities1 (Set[Entity]): A set of Entity objects from dataset 1.
        include_keys (bool): Include keys in tokenization.

    Returns:
        Dict[str, tuple[Set[int], Set[int]]]: A dictionary mapping tokens to tuples of sets of entity IDs.
        The first set contains the entitiy ids with this token from dataset 0, the second set contains the entity ids with this token from dataset 1.
    """
    blocks: Dict[str, tuple[Set[int], Set[int]]] = {}
    for dataset_id, entities in [(0, entities0), (1, entities1)]:
        for e in entities:
            for t in tokens(e, include_keys):
                blocks.setdefault(t, (set(), set()))
                blocks[t][dataset_id].add(e.id)
    for t in list(blocks.keys()):
        if len(blocks[t][0]) == 0 or len(blocks[t][1]) == 0:
            del blocks[t]
    return blocks


def clean_block_purging(
    blocks: Dict[str, tuple[Set[int], Set[int]]], prune_factor
) -> Dict[str, tuple[Set[int], Set[int]]]:
    """
    Filter out blocks based on a pruning factor and the maximum block size.

    Args:
        blocks (Dict[str, tuple[Set[int], Set[int]]]): A dictionary mapping tokens to tuples of sets of entity IDs.
        prune_factor: The pruning factor.

    Returns:
        Dict[str, tuple[Set[int], Set[int]]]: Filtered blocks.
    """
    bmax = clean_block_statistics(blocks)["max_size"]
    # Filter out blocks with size > prune_factor * bmax
    filtered_blocks = {
        key: value
        for key, value in blocks.items()
        if len(value[0]) + len(value[1]) <= prune_factor * bmax
    }
    return filtered_blocks


def clean_comparisons(
    blocks: Dict[str, tuple[Set[int], Set[int]]]
) -> Iterable[tuple[int, int]]:
    """
    Generate comparisons from blocks produced by clean_token_blocking.

    Args:
        blocks (Dict[str, tuple[Set[int], Set[int]]]): A dictionary mapping tokens to tuples of sets of entity IDs.

    Returns:
        Iterable[tuple[int, int]]: An iterable of entity ID pairs.
    """
    for blockpair in blocks.values():
        for e0, e1 in itertools.product(blockpair[0], blockpair[1]):
            yield e0, e1


def clean_block_statistics(blocks: Dict[str, tuple[Set[int], Set[int]]]) -> Dict:
    """
    Print statistics about blocks produced by clean_token_blocking.

    Args:
        blocks (Dict[str, tuple[Set[int], Set[int]]]): A dictionary mapping tokens to tuples of sets of entity IDs.

    Returns:
        Dict: A dictionary containing statistics.
    """
    # print statistics about the blocks
    sizes = list(
        map(lambda blockpair: len(blockpair[0]) + len(blockpair[1]), blocks.values())
    )
    return {
        "n": len(blocks),
        "mean_size": sum(sizes) / len(blocks),
        "min_size": min(sizes),
        "max_size": max(sizes),
    }


def overlap_coefficient(s1: Set, s2: Set) -> float:
    """
    Calculate the overlap coefficient between two sets.

    Args:
        s1 (Set): The first set.
        s2 (Set): The second set.

    Returns:
        float: The overlap coefficient.
    """
    if len(s1) == 0 or len(s2) == 0:
        if len(s1) == len(s2):
            return 1
        return 0
    return len(s1.intersection(s2)) / min(len(s1), len(s2))


if __name__ == "__main__":
    # experimental code for testing
    N = 100
    random_matches = set(get_random_matches(N))
    ids_0 = set()
    ids_1 = set()
    entities0 = set()
    entities1 = set()
    for m in random_matches:
        ids_0.add(m[0])
        ids_1.add(m[1])
        entities0.add(get_entity_by_id(m[0], "dbpedia0"))
        entities1.add(get_entity_by_id(m[1], "dbpedia1"))
    print(f"Token blocking on {2*N} entities with {N} matches")
    token_blocks = clean_token_blocking(entities0, entities1)
    cmps = set(clean_comparisons(token_blocks))
    print(f"Number of comparisons: ", len(cmps))
    # intersection of cmps with random_matches
    print(cmps.update(random_matches))
    print(f"Number of comparisons with all matches: ", len(cmps))
    clean_block_statistics(token_blocks)
    wstok = WhitespaceTokenizer(return_set=True)
    oc_nonmatch = []
    oc_match = []
    for e0, e1 in cmps:
        t0 = tokens(get_entity_by_id(e0, "dbpedia0"))
        t1 = tokens(get_entity_by_id(e1, "dbpedia1"))
        oc = overlap_coefficient(t0, t1)
        if (e0, e1) in random_matches:
            oc_match.append(oc)
        else:
            oc_nonmatch.append(oc)
    print(oc_match, oc_nonmatch)
    m = pd.DataFrame(oc_match)
    nm = pd.DataFrame(oc_nonmatch)
    print(m.describe())
    print(nm.describe())
