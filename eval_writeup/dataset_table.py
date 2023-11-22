import pandas as pd

# Load DataFrame from CSV
csv_filename = "eval_writeup/dataset_statistics.csv"
df = pd.read_csv(csv_filename)
# Remove prefixes and create a new column
df["Type"] = df["Dataset"].str.split("_").str[0]

# Remove prefixes from Dataset column
df["Dataset"] = df["Dataset"].str.replace(r"^structured_|textual_", "", regex=True)

# Reorder columns for clarity
df = df[["Type", "Dataset", "Total Pairs", "Positive Pairs", "Negative Pairs"]]
df["Dataset"] = df["Dataset"].str.replace("_", "-").str.title()
df["Dataset"] = df["Dataset"].str.replace("-1250", " Sampled")
# Create a new DataFrame with rows ending with "_1250"
# df_1250 = df[df["Dataset"].str.endswith("1250")]

# Display the new DataFrame
# print(df_1250)
# Display the loaded DataFrame
print(df)
df.to_csv("eval_writeup/ds_stats_table.csv")
