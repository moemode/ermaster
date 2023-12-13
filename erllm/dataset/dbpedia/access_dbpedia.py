import sqlite3
import json
from typing import Any, List
from erllm import DBFILE_PATH
from erllm.dataset.entity import Entity


def get_matches(N: int) -> List[Any]:
    match_query = f"""
    SELECT e0.uri, e0.kv, e1.uri, e1.kv from dbpedia0 AS e0, dbpedia1 AS e1, dbpedia_matches AS matches
    WHERE e0.id = matches.id0
    AND e1.id = matches.id1
    LIMIT {N};
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(match_query)
    res = cursor.fetchall()
    conn.close()
    return res


def get_entity_by_id(id: int, table: str) -> Entity:
    conn = sqlite3.connect(DBFILE_PATH)
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
        raise ValueError(f"Entity with id {id} not found in table {table}")


def get_random_matches(n: int):
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM dbpedia_matches ORDER BY RANDOM() LIMIT {n}")
    random_matches = cursor.fetchall()
    conn.close()
    return random_matches


def is_match(id0: int, id1: int) -> bool:
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM dbpedia_matches WHERE id0 = ? AND id1 = ?", (id0, id1)
    )
    match_count = cursor.fetchone()[0]
    conn.close()

    return match_count > 0


def get_number_of_entries(table: str) -> int:
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    conn.close()
    return count
