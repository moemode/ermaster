import pandas as pd
# Assuming your CSV file is named 'beer_data.csv'
a_path = 'beer_exp_data/exp_data/tableA.csv'
b_path = 'beer_exp_data/exp_data/tableB.csv'
# Read the first row of the CSV file to get the column names
columns = pd.read_csv(a_path, nrows=0).columns.tolist()
# Load the CSV into a Pandas DataFrame
ta = pd.read_csv(a_path, names=columns, skiprows=1)
tb = pd.read_csv(b_path, names=columns, skiprows=1)
# Assuming your CSV files are named 'test.csv', 'train.csv', and 'valid.csv'
truth_paths = ['beer_exp_data/exp_data/test.csv', 'beer_exp_data/exp_data/train.csv', 'beer_exp_data/exp_data/valid.csv']
# List to store DataFrames
dfs = []
# Loop through each file path and read the CSV into a DataFrame
for file_path in truth_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)
# Concatenate the DataFrames into one
matches = pd.concat(dfs, ignore_index=True)
print(ta,tb, matches)
