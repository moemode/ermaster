from typing import Iterable
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from writeup_utils import rename_datasets


def make_table(
    path,
    column_order,
    sort_by=None,
    reverse_color_columns: Iterable = [],
    color_columns: Iterable = [],
    rename=False,
    decimals=3,
    sort_ascending=False,
):
    # Read CSV data into a DataFrame
    df = pd.read_csv(path)

    if rename:
        df = rename_datasets(df, preserve_sampled=False)

    # Sort DataFrame by F1 column in descending order
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=sort_ascending)

    # Reorder columns
    df = df[["Dataset"] + column_order]

    def original_to_color(
        value, min_val, max_val, do_color: bool, decimals, reverse_coloring: bool
    ):
        high_color = "orange" if reverse_coloring else "cyan"
        low_color = "cyan" if reverse_coloring else "orange"
        r = (value - min_val) / (max_val - min_val)
        scaled_value = 2 * (r - 0.5)
        color = high_color if scaled_value > 0 else low_color
        color_intensity = abs(int(scaled_value * 50))

        prefix = f"\\cellcolor{{{color}!{color_intensity}}}" if do_color else ""

        # Ensure at least 'decimals' decimal places with trailing zero padding
        formatted_value = f"{value:.{decimals}f}"
        return f"{prefix}{formatted_value}"
        # return f"{prefix}{value:.{decimals}f}"
        # return f"{prefix}{round(value, decimals)}"

    # Apply color coding to each cell in the DataFrame
    df_colored = df.copy()
    for col in df.columns[1:]:
        df_colored[col] = df[col].apply(
            lambda x: original_to_color(
                x,
                df[col].min(),
                df[col].max(),
                col in color_columns,
                decimals,
                col in reverse_color_columns,
            )
        )

    # Convert DataFrame to LaTeX table
    latex_table = tabulate(
        df_colored,
        headers="keys",
        tablefmt="latex_raw",
        showindex=False,
        disable_numparse=True,
    )

    # Print the LaTeX table
    print(latex_table)


CONFIGURATIONS = {
    "base": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_selected.csv"),
        "color_columns": ["F1", "Precision", "Recall"],
    },
    "base_attr": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_wattr_names_selected.csv"),
        "color_columns": ["F1", "Precision", "Recall"],
    },
    "base_attr_rnd_order": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_wattr_names_rnd_order_selected.csv"),
        "color_columns": ["F1", "Precision", "Recall"],
    },
    "base_attr_embed_05": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_wattr_names_embed_05_selected.csv"),
        "color_columns": ["F1", "Precision", "Recall"],
    },
    "base_attr_embed_one_ppair": {
        "column_order": ["F1", "Precision", "Recall"],
        "sort_by": "F1",
        "path": Path("eval_writeup/base_wattr_names_embed_one_ppair_selected.csv"),
        "color_columns": ["F1", "Precision", "Recall"],
    },
    "base_hash": {
        "column_order": [
            "F1_Diff",
            "Precision_Diff",
            "Recall_Diff",
        ],
        "color_columns": ["F1_Diff", "Precision_Diff", "Recall_Diff"],
        "sort_by": "F1_Diff",
        "path": Path("eval_writeup/base_vs_hash.csv"),
    },
    "base_calibration": {
        "column_order": ["ECE", "ACE", "ACE-T", "ACE-F", "Brier Score"],
        "sort_by": "ECE",
        "path": Path("eval/calibration/base_calibration.csv"),
        "color_columns": ["ECE", "Brier Score"],
        "reverse_color_columns": ["ECE", "ACE", "ACE-T", "ACE-F", "Brier Score"],
        "sort_ascending": True,
        "rename": True,
    },
    "discarding_matcher_tradeoff": {
        "path": Path("eval_writeup/discarding_matcher_tradeoff_pivot.csv"),
        "color_columns": ["2.5", "5.0", "10.0"],
        "column_order": ["2.5", "5.0", "10.0"],
        "sort_by": "2.5",
        "decimals": 2,
    },
    "discarding_matcher_tradeoff_time": {
        "path": Path("eval_writeup/discarding_matcher_tradeoff_pivot_time.csv"),
        "color_columns": ["2.5", "5.0", "10.0"],
        "column_order": ["2.5", "5.0", "10.0"],
        "sort_by": "2.5",
        "decimals": 2,
    },
}
if __name__ == "__main__":
    cfg_name = "discarding_matcher_tradeoff_time"
    cfg_name = "base_attr_embed_one_ppair"
    cfg_name = "base_hash"
    make_table(**CONFIGURATIONS[cfg_name])

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
