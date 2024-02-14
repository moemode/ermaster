"""
Loads data from .txt file and loads it into SQLite tables. 
The primary tables store DBpedia entities with key-value pairs, and an additional table stores matching pairs.
"""

"""
Raw File Structure:

- The files `cleanDBPedia1out`, `cleanDBPedia2out` contain the entities.
Each line corresponds to a different entity profile and has the following structure (where n is number of attributes, aname and aval are the attribute names and values):  
`numerical_id , uri , n ,  aname_0 , aval_0 , aname_1 , aval_1 ,...aname_n , aval_n`

That is, the separator is `space,space`.
`,` in the original data have been replaced with `,,`.
This must be accounted for when reading the data.

- The file `newDBPediaMatchesout` contains matching profile pairs.
Each line has the format:  
`numerical_id_0 , numerical_id_1`

Table Structure:
- Entities Table (e.g., dbpedia0, dbpedia1):
  - Columns: id (INTEGER PRIMARY KEY), uri (TEXT), kv (JSON)

- Matches Table (e.g., dbpedia_matches):
  - Columns: id0 (INTEGER), id1 (INTEGER)
"""

from pathlib import Path
import json
from tqdm import tqdm
import sqlite3
from typing import Iterable, Tuple
from erllm import DBFILE_PATH, DBPEDIA_RAW_FOLDER_PATH

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


def db_execute(stmt: str):
    """
    Execute a SQL statement on the SQLite database.

    Args:
        stmt (str): The SQL statement to execute.
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    cursor.execute(stmt)
    conn.commit()
    conn.close()


def list_to_pairs(l: list) -> Iterable[Tuple[str, str]]:
    """
    Breakup list into pairs of length two.

    Args:
        l (list): The list to convert.

    Returns:
        Iterable[Tuple[str, str]]: A generator of pairs.
    """
    return ((l[i], l[i + 1]) for i in range(0, len(l), 2))


def load_dbpedia_entities(fname: Path, tname: str):
    """
    Load DBpedia entity data from a file into the specified table.

    Args:
        fname (Path): Path to the file containing DBpedia entity data.
        tname (str): Name of the table to load the data into.
    """
    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    # Define the SQL query with placeholders
    insert_query = f"INSERT INTO {tname} (id, uri, kv) VALUES (?, ?, ?)"
    # Execute the query with the data tuple
    with open(fname, "r") as f:
        # destructure each raw line into id, uri, n_kv, *kv
        for line_number, l in enumerate(tqdm(f, desc=f"Load table {tname}")):
            values = [item.strip() for item in l.split(" , ")]
            # during serialization ',' were replaced with ',,'. Undo this.
            values = [s.replace(",,", ",") for s in values]
            id, uri, n_kv, *kv = values
            id = int(id)
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
    """
    Load matching pairs data from a file into the specified table.

    Args:
        fname (Path): Path to the file containing matching pairs data.
        tname (str): Name of the table to load the data into.
    """
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
    """
    Load data into DBpedia tables and matching pairs table if provided.

    Args:
        dbpaths: Paths to DBpedia entity data files.
        dbtnames: Names of the DBpedia entity tables.
        matchpath: Path to the matching pairs data file.
        matchtname: Name of the matching pairs table.
    """
    for f, tname in zip(dbpaths, dbtnames):
        db_execute(create_dbpediadb_table_stmt.format(tname))
        load_dbpedia_entities(f, tname)
    if not matchpath:
        return
    db_execute(create_match_table_stmt.format(matchtname))
    load_dbpedia_matches(matchpath, matchtname)


def create_indices():
    """
    Create an index on the id0 column of the matches table.
    """

    conn = sqlite3.connect(DBFILE_PATH)
    cursor = conn.cursor()
    create_index_sql = "CREATE INDEX idx_id0 ON dbpedia_matches (id0);"
    cursor.execute(create_index_sql)
    conn.commit()
    print("Index created successfully.")


if __name__ == "__main__":
    dbpedia_raw_dir = DBPEDIA_RAW_FOLDER_PATH
    dbpaths = [
        dbpedia_raw_dir / "cleanDBPedia1out",
        dbpedia_raw_dir / "cleanDBPedia2out",
    ]
    tnames = ["dbpedia0", "dbpedia1"]
    matchpath = dbpedia_raw_dir / "newDBPediaMatchesout"
    load_dbpedia(dbpaths, tnames, matchpath, "dbpedia_matches")
    create_indices()
