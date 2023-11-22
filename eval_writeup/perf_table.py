import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Read CSV data into a DataFrame
df = pd.read_csv("eval_writeup/base_selected.csv")

# Scale and demean each column
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])


# Convert original values to LaTeX color codes
def original_to_color(value, mean, std):
    scaled_value = (value - mean) / std
    color = "cyan" if scaled_value > 0 else "orange"
    return f"\\cellcolor{{{color}!{abs(int(scaled_value*50))}}}{value:.3f}"


# Apply color coding to each cell in the DataFrame
df_colored = df.copy()
for col in df.columns[1:]:
    df_colored[col] = df[col].apply(
        lambda x: original_to_color(x, df[col].mean(), df[col].std())
    )

# Convert DataFrame to LaTeX table
latex_table = tabulate(
    df_colored, headers="keys", tablefmt="latex_raw", showindex=False
)

# Print the LaTeX table
print(latex_table)

"""
# Working code but has scaled values in table - not good
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


# Read CSV data into a DataFrame
df = pd.read_csv("eval_writeup/base_selected.csv")
# Scale and demean each column
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])


# Convert scaled values to LaTeX color codes
def scale_to_color(value):
    color = "cyan" if value > 0 else "orange"
    return f"\\cellcolor{{{color}!{abs(int(value*50))}}}{value:.3f}"


# Apply color coding to each cell in the DataFrame
df_colored = df_scaled.applymap(scale_to_color)

# Add dataset column to the colored DataFrame
df_colored.insert(0, "Dataset", df["Dataset"])

# Convert DataFrame to LaTeX table
latex_table = tabulate(
    df_colored, headers="keys", tablefmt="latex_raw", showindex=False
)

# Print the LaTeX table
print(latex_table)
"""
