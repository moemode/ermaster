import itertools
from typing import Dict, Iterable, Set
from access_dbpedia import Entity, get_entity_by_id, get_random_matches, tokens


def token_blocking(entities: Set[Entity]) -> Dict[str, Set[int]]:
    blocks: Dict[str, Set[int]] = {}
    for e in entities:
        for t in tokens(e):
            if t not in blocks:
                blocks[t] = set()
            blocks[t].add(e.id)
    # remove all blocks with only one entity
    for k in list(blocks.keys()):
        if len(blocks[k]) < 2:
            del blocks[k]
    return blocks


def block_statistics(blocks: Dict[str, Set[int]]):
    # print statistics about the blocks
    print(f"Number of blocks: {len(blocks)}")
    print(f"Average block size: {sum(map(len, blocks.values())) / len(blocks)}")
    print(f"Maximum block size: {max(map(len, blocks.values()))}")
    print(f"Minimum block size: {min(map(len, blocks.values()))}")


def comparisons(blocks: Dict[str, Set[int]]) -> Iterable[tuple[int, int]]:
    for b in blocks.values():
        for e0, e1 in itertools.combinations(b, 2):
            yield e0, e1


if __name__ == "__main__":
    N = 10
    random_matches = get_random_matches(N)
    ids_0 = set()
    ids_1 = set()
    entities = set()
    for m in random_matches:
        ids_0.add(m[0])
        ids_1.add(m[1])
        entities.add(get_entity_by_id(m[0], "dbpedia0"))
        entities.add(get_entity_by_id(m[1], "dbpedia1"))
    print(
        f"Token blocking on {len(entities)} entities with {int(len(entities)/2)} matches"
    )
    token_blocks = token_blocking(entities)
    block_statistics(token_blocks)
    print("Number of comparisons: ", len(list(comparisons(token_blocks))))
