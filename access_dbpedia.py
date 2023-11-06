import sqlite3
import json

DBFILE = "my_database.db"
N = 10  # Replace 10 with the number of random entries you want

class Entity(dict):
    def __init__(self, id, uri, kv):
        super().__init__(kv)
        self.id = id
        self.uri = uri


def get_matches(N):
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

def get_entity_by_id(id, table):
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

def get_random_matches(n):
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM dbpedia_matches ORDER BY RANDOM() LIMIT {n}")
    random_matches = cursor.fetchall()
    conn.close()
    return random_matches


if __name__ == "__main__":
    random_matches = get_random_matches(N)
    ids_0 = set()
    ids_1 = set()
    print(f"Random {N} Matches:")
    for m in random_matches:
        print(m)
        ids_0.add(m[0])
        ids_1.add(m[1])
    print(ids_0, ids_1)
    i = ids_0.pop()
    e = get_entity_by_id(i, "dbpedia0")
    print(e)
