from pathlib import Path
import pandas as pd
from typing import List, Tuple

from access_dbpedia import Entity, tokens


fpath = Path(
    "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets/structured_itunes_amazon"
)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(fpath / "test.csv")
df.set_index("_id", inplace=True)

# Display the first few rows of the DataFrame
print(df.head())
print(df.columns)
# Iterate over the rows of the DataFrame
table1_columns = [col for col in df.columns if col.startswith("table1")]
column_names = [col.split(".")[1] for col in table1_columns]
table2_columns = [col for col in df.columns if col.startswith("table2")]
assert list(map(lambda n: "table2." + n, column_names)) == table2_columns
columns_no_id = [name for name in column_names if name != "id"]
table1_columns_no_id = [col for col in table1_columns if col != "table1.id"]
table2_columns_no_id = [col for col in table2_columns if col != "table2.id"]

table1_entities = []
table2_entities = []
pairs: List[Tuple[bool, Entity, Entity]] = []
for _, row in df.iterrows():
    # Filter columns that start with "table1"
    # Extract values from the filtered columns and save them as a tuple
    kv1 = dict(zip(columns_no_id, row[table1_columns_no_id].values))
    kv2 = dict(zip(columns_no_id, row[table2_columns_no_id].values))
    table1_tuple = zip(table1_columns_no_id, row[table1_columns_no_id].values)
    table2_tuple = zip(table2_columns_no_id, row[table2_columns_no_id])
    table1_entities.append(Entity(row["table1.id"], None, kv1, order=columns_no_id))
    print(tokens(table1_entities[-1]))
    print(tokens(table1_entities[-1], include_keys=False))
    table2_entities.append(Entity(row["table2.id"], None, kv2, order=columns_no_id))
print(table1_entities[:5])
