from collections import OrderedDict
import sqlite3
import json
import re
from typing import Any, Iterable, List, Optional, Set, Tuple


DBFILE = "my_database.db"


class Entity(dict):
    def __init__(self, id, uri, kv, order=None):
        super().__init__(kv)
        self.id = id
        self.uri = uri
        self.order = order
        if order and len(order) != len(self.keys()):
            raise ValueError("order must contain all keys")

    # make hashable
    def __hash__(self):
        return hash(self.id)


class OrderedEntity(OrderedDict):
    def __init__(
        self, id: int, uri: Optional[str], attributes: Iterable[Tuple[str, str]]
    ):
        super().__init__(attributes)
        self.id = id
        self.uri = uri

    # make hashable
    def __hash__(self) -> int:
        return hash(self.id)

    def tokens(self, include_keys=False, return_set=True):
        if include_keys:
            it = (f"{k} {v}" for (k, v) in self.items())
        else:
            it = self.values()
        vals = " ".join(it).lower()
        toks = filter(None, re.split("[\\W_]", vals))
        return set(toks) if return_set else list(toks)

    def value_string(self) -> str:
        return " ".join(str(value) for value in self.values())


def tokens(
    e: Entity,
    include_keys=False,
    return_set=True,
) -> Set[str] | List[str]:
    order = e.order
    if order is None:
        order = sorted(e.keys())
    if include_keys:
        it = (f"{key} {e[key]}" for key in order)
    else:
        it = (e[key] for key in order)
    vals = " ".join(it).lower()
    toks = filter(None, re.split("[\\W_]", vals))
    return set(toks) if return_set else list(toks)


def get_matches(N: int) -> List[Any]:
    match_query = f"""
    SELECT e0.uri, e0.kv, e1.uri, e1.kv from dbpedia0 AS e0, dbpedia1 AS e1, dbpedia_matches AS matches
    WHERE e0.id = matches.id0
    AND e1.id = matches.id1
    LIMIT {N};
    """
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    cursor.execute(match_query)
    res = cursor.fetchall()
    conn.close()
    return res


def get_entity_by_id(id: int, table: str) -> Optional[Entity]:
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    # Define the SQL query to retrieve an entry by its id
    cursor.execute(f"SELECT id, uri, kv FROM {table} WHERE id = ?", (id,))
    entry = cursor.fetchone()  # Fetch the first matching entry
    conn.close()
    if entry:
        id, uri, kv_json = entry
        kv = json.loads(kv_json)
        return Entity(id, uri, kv)
    else:
        return None


def get_random_matches(n: int):
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM dbpedia_matches ORDER BY RANDOM() LIMIT {n}")
    random_matches = cursor.fetchall()
    conn.close()
    return random_matches
