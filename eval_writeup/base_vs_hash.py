import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from writeup_utils import rename_datasets

import pandas as pd

# Read in the CSV file into a DataFrame
df = pd.read_csv(
    "eval/llm_matcher/base_hash.csv"
)  # Replace "your_file.csv" with the path to your CSV file

metrics_df = df[
    [
        "Dataset",
        "PromptType",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
    ]
]

# Pivot the table for comparison
comparison_pivot = metrics_df.pivot_table(index="Dataset", columns="PromptType")


comparison_pivot["Recall_Diff"] = (
    comparison_pivot["Recall"]["general_complex_force_hash"]
    - comparison_pivot["Recall"]["general_complex_force"]
)
comparison_pivot["Precision_Diff"] = (
    comparison_pivot["Precision"]["general_complex_force_hash"]
    - comparison_pivot["Precision"]["general_complex_force"]
)
comparison_pivot["F1_Diff"] = (
    comparison_pivot["F1"]["general_complex_force_hash"]
    - comparison_pivot["F1"]["general_complex_force"]
)
# Calculate the difference between PromptTypes for each metric
comparison_pivot["Accuracy_Diff"] = (
    comparison_pivot["Accuracy"]["general_complex_force_hash"]
    - comparison_pivot["Accuracy"]["general_complex_force"]
)

df = comparison_pivot.reset_index(drop=False)
# make dataset column index
df = df[
    [
        "Dataset",
        "F1_Diff",
        "Recall_Diff",
        "Precision_Diff",
    ]
]
df = df.set_index("Dataset")
df.columns = df.columns.get_level_values(0)
df["F1 base"] = comparison_pivot["F1"]["general_complex_force"]
df["F1 hash"] = comparison_pivot["F1"]["general_complex_force_hash"]
# df = df[["F1 base", "F1 hash", "F1_Diff", "Recall_Diff", "Precision_Diff"]]
df = df.reset_index(drop=False)
df = rename_datasets(df, preserve_sampled=False)
df.to_csv("eval_writeup/base_vs_hash.csv", index=False)


# Rename 'general_complex_force' to an empty string and 'general_complex_force_hash' to 'hash'
metrics_df["PromptType"] = metrics_df["PromptType"].replace(
    {"general_complex_force": "base", "general_complex_force_hash": "hash"}
)
metrics_df = metrics_df.sort_values(by=["Dataset", "PromptType"])
metrics_df = rename_datasets(metrics_df, preserve_sampled=False)

# Create a pivot table with 'Dataset' as index and 'PromptType' as columns
pivot_metrics = pd.pivot_table(metrics_df, index="Dataset", columns="PromptType")
# remove Accuracy column
pivot_metrics = pivot_metrics.drop(columns="Accuracy", level=0)
pivot_metrics.columns.names = [None, None]
# sort in decreasing order of difference between F1 of hash - base
# Calculate the difference between F1 of 'hash' and 'base'
pivot_metrics["F1_Diff"] = pivot_metrics["F1", "hash"] - pivot_metrics["F1", "base"]
# Sort the pivot table based on the F1 difference in decreasing order
pivot_metrics_sorted = pivot_metrics.sort_values(by=("F1_Diff"), ascending=False)
pivot_metrics = pivot_metrics_sorted.drop(columns=("F1_Diff"))
modified_index = pd.MultiIndex.from_product(
    [["F1", "Precision", "Recall"], ["hash", "base"]]
)
pivot_metrics = pd.DataFrame(pivot_metrics, columns=modified_index)
s = pivot_metrics.style
s.format(precision=3)
latex_table = s.to_latex(
    "eval_writeup/base_vs_hash_abs.tex",
    # column_format="lccc",
    hrules=True,
    convert_css=True,
    position_float="centering",
    multicol_align="c",
    caption="F1, precision and recall for LLM Matcher (gpt-3.5-turbo-instruct) using base and hash prompt design.",
    label="tab:base-vs-hash-abs",
)
# Flatten the multi-level columns and rename based on PromptType
# pivot_metrics.columns = [f"{col[1]}_{col[0]}" for col in pivot_metrics.columns]

# Display the renamed pivot table
# print(pivot_metrics)
"""
        pivot_table = results.pivot_table(
            index="Dataset", columns=["Method"], values=cfg["columns"]
        )
        pivot_table = pivot_table.reindex(["SM", "Ditto"], axis=1, level=1)
        # order decreasingly in terms of F1 of SM
        pivot_table = pivot_table.sort_values(by=("F1", "SM"), ascending=False)
        # Add a row  "All" with the mean of the F1, Precision and Recall
        pivot_table.loc["All"] = pivot_table.mean()
        s = pivot_table.style
        s.format(precision=2)
        latex_table = s.to_latex(
            EVAL_FOLDER_PATH / "ditto" / f"{cfg_name}_comparison_table.tex",
            hrules=True,
            position_float="centering",
            multicol_align="c",
            caption=f"Comparison of classification performance",
        )
"""
