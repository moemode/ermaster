import pandas as pd

# Read in the CSV file into a DataFrame
df = pd.read_csv(
    "eval_writeup/hash_vs_non.csv"
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
    comparison_pivot["Recall"]["general_complex_force"]
    - comparison_pivot["Recall"]["general_complex_force_hash"]
)
comparison_pivot["Precision_Diff"] = (
    comparison_pivot["Precision"]["general_complex_force"]
    - comparison_pivot["Precision"]["general_complex_force_hash"]
)
comparison_pivot["F1_Diff"] = (
    comparison_pivot["F1"]["general_complex_force"]
    - comparison_pivot["F1"]["general_complex_force_hash"]
)
# Calculate the difference between PromptTypes for each metric
comparison_pivot["Accuracy_Diff"] = (
    comparison_pivot["Accuracy"]["general_complex_force"]
    - comparison_pivot["Accuracy"]["general_complex_force_hash"]
)

print(comparison_pivot[["Accuracy_Diff", "Recall_Diff", "Precision_Diff", "F1_Diff"]])

# Rename 'general_complex_force' to an empty string and 'general_complex_force_hash' to 'hash'
metrics_df["PromptType"] = metrics_df["PromptType"].replace(
    {"general_complex_force": "base", "general_complex_force_hash": "hash"}
)
metrics_df = metrics_df.sort_values(by=["Dataset", "PromptType"])
print(metrics_df)

# Create a pivot table with 'Dataset' as index and 'PromptType' as columns
pivot_metrics = pd.pivot_table(metrics_df, index="Dataset", columns="PromptType")

# Flatten the multi-level columns and rename based on PromptType
pivot_metrics.columns = [f"{col[1]}_{col[0]}" for col in pivot_metrics.columns]

# Display the renamed pivot table
# print(pivot_metrics)
