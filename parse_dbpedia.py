from pathlib import Path
from itertools import pairwise
import json
import tqdm
from tqdm import tqdm
import sqlite3

DBFILE = "my_database.db"

create_dbpediadb_table_stmt = """
        CREATE TABLE IF NOT EXISTS {} (
            id INTEGER PRIMARY KEY,
            uri TEXT,
            kv JSON
        )
        """

create_match_table_stmt = """
        CREATE TABLE IF NOT EXISTS {} (
            id0 INTEGER,
            id1 INTEGER
        )
        """


def db_execute(stmt):
    # Connect to the SQLite database or create it if it doesn't exist
    conn = sqlite3.connect(DBFILE)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Create a table with the specified schema
    cursor.execute(stmt)
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def load_dbpedia_db(fname: Path, tname: str):
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    # Define the SQL query with placeholders
    insert_query = f"INSERT INTO {tname} (id, uri, kv) VALUES (?, ?, ?)"
    # Execute the query with the data tuple
    with open(fname, "r") as f:
        for line_number, l in enumerate(tqdm(f)):
            values = [item.strip() for item in l.split(" , ")]
            values = [s.replace(",,", ",") for s in values]
            id, uri, n_kv, *kv = values
            id = int(id)
            n_kv = int(n_kv)
            kv_json = json.dumps(dict(pairwise(kv[:-1])))
            assert int(id) == line_number
            assert (
                len(kv) == 2 * int(n_kv) + 1
            )  # trailing ' , ' gives extra element + 1
            cursor.execute(insert_query, (id, uri, kv_json))
    conn.commit()
    conn.close()


def load_dbpedia_matches(fname: Path, tname: str):
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    # Define the SQL query with placeholders
    insert_query = f"INSERT INTO {tname} (id0, id1) VALUES (?, ?)"
    # Execute the query with the data tuple
    with open(fname, "r") as f:
        for l in tqdm(f):
            values = tuple(item.strip() for item in l.split(" , "))
            assert len(values) == 2
            cursor.execute(insert_query, values)
    conn.commit()
    conn.close()


def load_dbpedia(dbpaths, dbtnames, matchpath=None, matchtname=None):
    for f, tname in zip(dbpaths, dbtnames):
        db_execute(create_dbpediadb_table_stmt.format(tname))
        print(f"Load table {tname}")
        load_dbpedia_db(f, tname)
    if not matchpath:
        return
    db_execute(create_match_table_stmt.format(matchtname))
    print(f"Load table {matchtname}")
    load_dbpedia_matches(matchpath, matchtname)


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


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    dbpaths = [current_dir / "cleanDBPedia1out", current_dir / "cleanDBPedia2out"]
    tnames = ["dbpedia0", "dbpedia1"]
    matchpath = current_dir / "newDBPediaMatchesout"
    load_dbpedia(dbpaths, tnames, matchpath, "dbpedia_matches")
