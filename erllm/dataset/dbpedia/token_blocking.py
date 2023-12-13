from typing import Dict, Set, Iterable
from .access_dbpedia import (
    Entity,
    get_entity_by_id,
    get_random_matches,
    tokens,
)
import itertools
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd


def token_blocking(entities: Set[Entity]) -> Dict[str, Set[int]]:
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
    for b in blocks.values():
        for e0, e1 in itertools.combinations(b, 2):
            yield e0, e1


def block_statistics(blocks: Dict[str, Set[int]]):
    print(f"Number of blocks: {len(blocks)}")
    print(f"Average block size: {sum(map(len, blocks.values())) / len(blocks)}")
    print(f"Maximum block size: {max(map(len, blocks.values()))}")
    print(f"Minimum block size: {min(map(len, blocks.values()))}")


def clean_token_blocking(
    entities0: Set[Entity], entities1: Set[Entity], include_keys=False
) -> Dict[str, tuple[Set[int], Set[int]]]:
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


def clean_block_purging(blocks: Dict[str, tuple[Set[int], Set[int]]], prune_factor):
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
    for blockpair in blocks.values():
        for e0, e1 in itertools.product(blockpair[0], blockpair[1]):
            yield e0, e1


def clean_block_statistics(blocks: Dict[str, tuple[Set[int], Set[int]]]):
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


def overlap_coefficient(s1: Set, s2: Set):
    if len(s1) == 0 or len(s2) == 0:
        if len(s1) == len(s2):
            return 1
        return 0
    return len(s1.intersection(s2)) / min(len(s1), len(s2))


if __name__ == "__main__":
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
