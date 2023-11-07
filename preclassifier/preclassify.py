from pathlib import Path
import pandas as pd

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
field_names = [col.split(".")[1] for col in table1_columns]
table2_columns = [col for col in df.columns if col.startswith("table2")]
assert list(map(lambda n: "table2." + n, field_names)) == table2_columns
table1_entities = []
table2_entities = []
for _, row in df.iterrows():
    # Filter columns that start with "table1"
    table1_columns = [col for col in row.index if col.startswith("table1")]
    # Extract values from the filtered columns and save them as a tuple
    table1_tuple, table2_tuple = tuple(row[table1_columns]), tuple(row[table2_columns])
    # Append the tuple to the list
    row["label"] = 1
    table1_entities.append(table1_tuple)
print(table1_entities[:5])
