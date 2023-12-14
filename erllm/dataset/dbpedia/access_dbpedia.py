import sqlite3
import json
from typing import Any, List
from erllm import DBFILE_PATH
from erllm.dataset.entity import Entity


def get_matches(N: int) -> List[Any]:
    """
    Retrieve a list of matching pairs with all entity attributes.

    Args:
        N (int): The maximum number of matching pairs to retrieve.

    Returns:
        List[Any]: A list of tuples containing matching entities from 'dbpedia0' and 'dbpedia1'.
    """
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
    """
    Retrieve an entity by its ID from the specified table.

    Args:
        id (int): The ID of the entity to retrieve.
        table (str): The name of the table containing the entity.

    Returns:
        Entity: An Entity object representing the retrieved entity.

    Raises:
        ValueError: If the entity with the specified ID is not found in the table.
    """
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


def get_random_matches(n: int) -> List[Any]:
    """
    Retrieve a list of random matching pair ids from the 'dbpedia_matches' table.

    Args:
        n (int): The number of random matching pairs to retrieve.

    Returns:
        List[Any]: A list of tuples containing the ids of random matching pairs.
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM dbpedia_matches ORDER BY RANDOM() LIMIT {n}")
    random_matches = cursor.fetchall()
    conn.close()
    return random_matches


def is_match(id0: int, id1: int) -> bool:
    """
    Check if the entities are a match by checking the 'dbpedia_matches' table.

    Args:
        id0 (int): The ID of the first entity.
        id1 (int): The ID of the second entity.

    Returns:
        bool: True if a matching pair exists, False otherwise.
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM dbpedia_matches WHERE id0 = ? AND id1 = ?", (id0, id1)
    )
    match_count = cursor.fetchone()[0]
    conn.close()

    return match_count > 0


def get_number_of_entries(table: str) -> int:
    """
    Get the total number of entries in the specified table.

    Args:
        table (str): The name of the table.

    Returns:
        int: The number of entries in the table.
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    conn.close()
    return count
