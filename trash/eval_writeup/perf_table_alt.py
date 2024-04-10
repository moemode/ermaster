import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from pathlib import Path


def make_table(path, sortby, column_order):
    # Read CSV data into a DataFrame
    df = pd.read_csv(path)
    # Sort DataFrame by F1 column in descending order
    df = df.sort_values(by=sortby, ascending=False)
    # Reorder columns
    df = df[["Dataset"] + column_order]

    # Scale and demean each column
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:], index=df.index
    )

    # Convert scaled values to LaTeX color codes
    def scale_to_color(value):
        color = "cyan" if value > 0 else "orange"
        return f"\\cellcolor{{{color}!{abs(int(value*50))}}}{value:.3f}"

    # Convert scaled values to LaTeX color codes
    def scale_to_color_no_val(value):
        color = "cyan" if value > 0 else "orange"
        return f"\\cellcolor{{{color}!{abs(int(value*50))}}}"

    def my_round(value):
        return f"{value:.3f}"

    df_rounded = df.copy()
    df_rounded = df_rounded.iloc[:, 1:].applymap(my_round)
    # Apply color coding to each cell in the DataFrame
    df_colored = df_scaled.applymap(scale_to_color_no_val)

    result_df = df_colored.astype(str) + df_rounded.astype(str)
    result_df.insert(0, "Dataset", df["Dataset"])
    # Add dataset column to the colored DataFrame
    df_colored.insert(0, "Dataset", df["Dataset"])

    # Convert DataFrame to LaTeX table
    latex_table = tabulate(
        result_df, headers="keys", tablefmt="latex_raw", showindex=False
    )

    # Print the LaTeX table
    print(latex_table)


CONFIGURATIONS = {
    "base": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_selected.csv"),
    },
    "base_hash": {
        "column_order": ["F1_Diff", "Precision_Diff", "Recall_Diff"],
        "sort_by": "F1_Diff",
        "path": Path("eval_writeup/base_vs_hash.csv"),
    },
}
if __name__ == "__main__":
    cfg = CONFIGURATIONS["base_hash"]
    make_table(cfg["path"], cfg["sort_by"], cfg["column_order"])
