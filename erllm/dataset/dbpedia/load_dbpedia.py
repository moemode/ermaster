from pathlib import Path
import json
from tqdm import tqdm
import sqlite3
from typing import Iterable, Tuple
from erllm.dataset import DBFILE_PATH

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
    conn = sqlite3.connect(DBFILE_PATH)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Create a table with the specified schema
    cursor.execute(stmt)
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def list_to_pairs(l: list) -> Iterable[Tuple[str, str]]:
    return ((l[i], l[i + 1]) for i in range(0, len(l), 2))


def load_dbpedia_db(fname: Path, tname: str):
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    # Define the SQL query with placeholders
    insert_query = f"INSERT INTO {tname} (id, uri, kv) VALUES (?, ?, ?)"
    # Execute the query with the data tuple
    with open(fname, "r") as f:
        for line_number, l in enumerate(tqdm(f, desc=f"Load table {tname}")):
            values = [item.strip() for item in l.split(" , ")]
            values = [s.replace(",,", ",") for s in values]
            id, uri, n_kv, *kv = values
            id = int(id)
            if id == 539677:
                print("hi")
            n_kv = int(n_kv)
            kv_dict = {}
            # Iterate through the list of tuples
            for key, value in list_to_pairs(kv[:-1]):
                # If the key is already in the dictionary, concatenate the values
                if key in kv_dict:
                    kv_dict[key] += " " + value
                else:
                    # If the key is not in the dictionary, add a new entry
                    kv_dict[key] = value
            kv_json = json.dumps(kv_dict)
            assert int(id) == line_number
            assert (
                len(kv) == 2 * int(n_kv) + 1
            )  # trailing ' , ' gives extra element + 1
            cursor.execute(insert_query, (id, uri, kv_json))
    conn.commit()
    conn.close()


def load_dbpedia_matches(fname: Path, tname: str):
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    # Define the SQL query with placeholders
    insert_query = f"INSERT INTO {tname} (id0, id1) VALUES (?, ?)"
    # Execute the query with the data tuple
    with open(fname, "r") as f:
        for l in tqdm(f, desc=f"Load table {tname}"):
            values = tuple(item.strip() for item in l.split(" , "))
            assert len(values) == 2
            cursor.execute(insert_query, values)
    conn.commit()
    conn.close()


def load_dbpedia(dbpaths, dbtnames, matchpath=None, matchtname=None):
    for f, tname in zip(dbpaths, dbtnames):
        db_execute(create_dbpediadb_table_stmt.format(tname))
        load_dbpedia_db(f, tname)
    if not matchpath:
        return
    db_execute(create_match_table_stmt.format(matchtname))
    load_dbpedia_matches(matchpath, matchtname)


def create_indices():
    # Connect to the SQLite database
    conn = sqlite3.connect(DBFILE_PATH)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Define the SQL command to create an index on the id0 column
    create_index_sql = "CREATE INDEX idx_id0 ON dbpedia_matches (id0);"
    cursor.execute(create_index_sql)
    # Commit the changes to the database
    conn.commit()
    print("Index created successfully.")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    dbpaths = [current_dir / "cleanDBPedia1out", current_dir / "cleanDBPedia2out"]
    tnames = ["dbpedia0", "dbpedia1"]
    matchpath = current_dir / "newDBPediaMatchesout"
    load_dbpedia(dbpaths, tnames, matchpath, "dbpedia_matches")
    create_indices()
