import pandas as pd

# Read in the CSV file into a DataFrame
df = pd.read_csv(
    "eval_writeup/base.csv"
)  # Replace "your_file.csv" with the path to your CSV file

metrics_df = df[
    [
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
    ]
]

print(metrics_df)
