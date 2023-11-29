import pandas as pd
from tabulate import tabulate
from writeup_utils import rename_datasets

# load eval_writeup/pre_llm_tradeoff.csv
df = pd.read_csv("eval_writeup/pre_llm_tradeoff.csv")
df = rename_datasets(df, preserve_sampled=False)
df = df[["Dataset", "F1_Decrease_Threshold", "F1_Decrease", "Cost Decrease"]]
df["F1_Decrease_Threshold"] = df["F1_Decrease_Threshold"].abs()

print(df)
# Format numerical columns as percentages with two decimals
df_formatted = df.style.format(
    {
        "F1_Decrease_Threshold": "{:.2%}",
        "F1_Decrease": "{:.2%}",
        "Cost Decrease": "{:.2%}",
    }
)
# df_formatted.hide(axis="index")

df_formatted.hide(axis="index").to_latex("eval_writeup/pre_llm_tradeoff_formatted.ltx")

# Format numerical columns as percentages with two decimals
df["F1_Decrease_Threshold"] = df["F1_Decrease_Threshold"].apply(lambda x: f"{x:.2%}")
df["F1_Decrease"] = df["F1_Decrease"].apply(lambda x: f"{x:.2%}")
df["Cost Decrease"] = df["Cost Decrease"].apply(lambda x: f"{x:.2%}")

# Print the nicely formatted table
table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
print(table)
