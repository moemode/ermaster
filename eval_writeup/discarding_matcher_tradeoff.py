import pandas as pd
from writeup_utils import rename_datasets

df = pd.read_csv("eval_writeup/discarding_matcher_tradeoff.csv")
df = rename_datasets(df, preserve_sampled=False)
df = df[
    [
        "Dataset",
        "F1_Decrease_Threshold",
        "F1_Decrease",
        "Cost Decrease",
        "Time Decrease",
    ]
]
df["F1_Decrease_Threshold"] = df["F1_Decrease_Threshold"].abs()

"""
# Format numerical columns as percentages with two decimals
df["F1_Decrease_Threshold"] = df["F1_Decrease_Threshold"].apply(lambda x: f"{x:.2%}")
df["F1_Decrease"] = df["F1_Decrease"].apply(lambda x: f"{x:.2%}")
df["Cost Decrease"] = df["Cost Decrease"].apply(lambda x: f"{x:.2%}")
df["Time Decrease"] = df["Time Decrease"].apply(lambda x: f"{x:.2%}")
df.to_latex("eval_writeup/discarding_matcher_tradeoff.ltx", escape=True, index=False)
"""
df["F1_Decrease_Threshold"] = (df["F1_Decrease_Threshold"] * 100).round(2)
df["F1_Decrease"] = (df["F1_Decrease"] * 100).round(2)
df["Cost Decrease"] = (df["Cost Decrease"] * 100).round(2)
df["Time Decrease"] = (df["Time Decrease"] * 100).round(2)
df.to_latex(
    "eval_writeup/discarding_matcher_tradeoff.ltx",
    escape=True,
    index=False,
    float_format="%.2f",
)

# Order the values in "F1_Decrease_Threshold" in ascending order
# order = sorted(df["F1_Decrease_Threshold"].unique(), key=lambda x: float(x.strip("%")))
"""
df["F1_Decrease_Threshold"] = pd.Categorical(
    df["F1_Decrease_Threshold"], categories=order, ordered=True
)
"""

# Pivot the DataFrame based on "F1_Decrease_Threshold" and show "Cost Decrease"
pivot_df = df.pivot_table(
    index="Dataset",
    columns="F1_Decrease_Threshold",
    values="Cost Decrease",
    aggfunc="first",
)

pivot_df.to_csv("eval_writeup/discarding_matcher_tradeoff_pivot.csv")


# Save the pivoted DataFrame to a LaTeX file
pivot_df.to_latex(
    "eval_writeup/discarding_matcher_tradeoff_pivot.ltx",
    escape=True,
    float_format="%.2f",
)

"""
# Print the nicely formatted table
table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
print(table)
"""
