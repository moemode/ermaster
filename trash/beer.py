from typing import List, Dict
import pandas as pd
import json
from pathlib import Path


def serialize(beer_entity: Dict, attributes: List) -> str:
    return " ".join([beer_entity[a] for a in attributes])


# Assuming your CSV file is named 'beer_data.csv'
a_path = "beer_exp_data/exp_data/tableA.csv"
b_path = "beer_exp_data/exp_data/tableB.csv"
# Read the first row of the CSV file to get the column names
attribute_names = pd.read_csv(a_path, nrows=0).columns.tolist()
serialize_attributes = [a for a in attribute_names if a != "id"]
# Load the CSV into a Pandas DataFrame
ta = pd.read_csv(a_path, names=attribute_names, skiprows=1)
tb = pd.read_csv(b_path, names=attribute_names, skiprows=1)
# Assuming your CSV files are named 'test.csv', 'train.csv', and 'valid.csv'
truth_paths = [
    "beer_exp_data/exp_data/test.csv",
    "beer_exp_data/exp_data/train.csv",
    "beer_exp_data/exp_data/valid.csv",
]
# List to store DataFrames
dfs = []
# Loop through each file path and read the CSV into a DataFrame
for file_path in truth_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)
# Concatenate the DataFrames into one
matches = pd.concat(dfs, ignore_index=True)
print(ta, tb, matches)

data = []
for i, row in matches.iterrows():  # merged_df.iterrows():
    truth = bool(row["label"])
    el = ta[ta["id"] == row.ltable_id].iloc[0].to_dict()
    er = tb[tb["id"] == row.rtable_id].iloc[0].to_dict()
    print(truth)
    sel = serialize(el, serialize_attributes)
    ser = serialize(er, serialize_attributes)
    data.append({"t": truth, "e0": sel, "e1": ser})

with open(Path("data") / "beer.json", "w") as json_file:
    json.dump(data, json_file, indent=2)
