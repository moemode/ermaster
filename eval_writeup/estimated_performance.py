import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
from writeup_utils import rename_datasets

# Read in the CSV file into a DataFrame
df = pd.read_csv("eval/llm_matcher/base.csv")

"""
df = df[
    [
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        # "Accuracy",
    ]
]
"""

df = rename_datasets(df, preserve_sampled=False)
# Compare Accuracy to EST_Accuracy
df["Accuracy_Difference"] = df["EST_Accuracy"] - df["Accuracy"]
df["Accuracy_Relative_Difference"] = (df["EST_Accuracy"] - df["Accuracy"]) / df[
    "Accuracy"
]

# Compare Precision to EST_Precision
df["Precision_Difference"] = df["EST_Precision"] - df["Precision"]
df["Precision_Relative_Difference"] = (df["EST_Precision"] - df["Precision"]) / df[
    "Precision"
]

# Compare Recall to EST_Recall
df["Recall_Difference"] = df["EST_Recall"] - df["Recall"]
df["Recall_Relative_Difference"] = (df["EST_Recall"] - df["Recall"]) / df["Recall"]

# Compare F1 to EST_F1
df["F1_Difference"] = df["EST_F1"] - df["F1"]
df["F1_Relative_Difference"] = (df["EST_F1"] - df["F1"]) / df["F1"]


# Create a new DataFrame with only the relevant columns
result_df = df[
    [
        "Dataset",
        # "Accuracy_Difference",
        "Accuracy_Relative_Difference",
        # "Precision_Difference",
        "Precision_Relative_Difference",
        # "Recall_Difference",
        "Recall_Relative_Difference",
        # "F1_Difference",
        "F1_Relative_Difference",
    ]
]

# Multiply all numerical columns by 100 (excluding "Dataset" column)
numeric_columns = result_df.select_dtypes(include="number").columns
result_df = result_df.round(4)
result_df[numeric_columns] = result_df[numeric_columns] * 100
# Save the pivoted DataFrame to a LaTeX file
result_df.to_latex(
    "eval_writeup/estimated_performance.ltx",
    escape=True,
    float_format="%.2f",
    index=False,
)
# Display the updated DataFrame
print(result_df)
# df.to_csv("eval_writeup/base_selected.csv", index=False)
