import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
from writeup_utils import rename_datasets

# Read in the CSV file into a DataFrame
df = pd.read_csv(
    "eval_writeup/base.csv"
)  # Replace "your_file.csv" with the path to your CSV file

df = df[
    [
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        # "Accuracy",
    ]
]

df = rename_datasets(df, preserve_sampled=False)
df.to_csv("eval_writeup/base_selected.csv", index=False)
