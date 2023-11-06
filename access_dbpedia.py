import sqlite3
import json
import re
from itertools import chain
from typing import Any, List, Optional, Set


DBFILE = "my_database.db"


class Entity(dict):
    def __init__(self, id, uri, kv):
        super().__init__(kv)
        self.id = id
        self.uri = uri

    # make hashable
    def __hash__(self):
        return hash(self.id)


def tokens(e: Entity, values_only=True) -> Set[str]:
    it = chain(e.values() if values_only else e.values(), e.keys())
    vals = " ".join(it).lower()
    return set(filter(None, re.split("[\\W_]", vals)))


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
