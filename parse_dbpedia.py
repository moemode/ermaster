from pathlib import Path
from itertools import pairwise
import json
import tqdm
from tqdm import tqdm
import sqlite3

DBFILE = "my_database.db"


def create_dbpedia_table(tname):
    # Connect to the SQLite database or create it if it doesn't exist
    conn = sqlite3.connect(DBFILE)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Create a table with the specified schema
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tname} (
            id INTEGER PRIMARY KEY,
            uri TEXT,
            kv JSON
        )
        """
    )
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def parse_dbpedia(fname: Path, tname: str):
    # Connect to the SQLite database
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


if __name__ == "__main__":
    tname = "dbpedia1"
    create_dbpedia_table(tname)
    current_dir = Path(__file__).resolve().parent
    # Create a Path object to the file in the same directory
    file_path = current_dir / "cleanDBPedia1out"
    parse_dbpedia(file_path, tname)
