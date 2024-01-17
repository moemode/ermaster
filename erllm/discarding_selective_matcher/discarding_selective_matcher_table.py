import pandas as pd
from erllm import EVAL_FOLDER_PATH

CONFIGURATIONS = {
    "basic-cmp": {
        "result_folder": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "basic_cmp",
    },
    "grid": {
        "result_folder": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "grid",
        "means": ["F1", "Precision", "Recall", "Accuracy"],
    },
}


def format_percentages(c):
    return f"{c*100:.0f}\%"


if __name__ == "__main__":
    cfg_name = "basic-cmp"
    cfg = CONFIGURATIONS[cfg_name]
    df = pd.read_csv(cfg["result_folder"] / "mean_f1.csv")
    # Pivot the DataFrame
    pivot_df = df.pivot(index="Discard Fraction", columns="Label Fraction", values="F1")
    # Format MultiIndex columns and index as percentages
    pivot_df.columns = pd.MultiIndex.from_tuples(
        [(pivot_df.columns.name, c) for c in pivot_df.columns]
    ).sort_values()
    pivot_df.index = pd.MultiIndex.from_tuples(
        [(pivot_df.index.name, c) for c in pivot_df.index]
    ).sort_values()
    first_row = pivot_df.index[0]
    first_col = pivot_df.columns[0]
    second_row = pivot_df.index[1]
    second_col = pivot_df.columns[1]
    idx = pd.IndexSlice
    slice_ = idx[idx[*first_row],]
    slice_first_cell = idx[idx[*first_row], idx[*first_col]]
    slice_first_row = idx[idx[*first_row], idx[second_col[0], second_col[1] :]]
    slice_first_col = idx[idx[second_row[0], second_row[1] :], idx[*first_col]]
    slice_rest = idx[
        idx[second_row[0], second_row[1] :], idx[second_col[0], second_col[1] :]
    ]
    s = pivot_df.style
    s.set_properties(**{"background-color": "lightgreen"}, subset=slice_first_row)
    s.set_properties(**{"background-color": "lightblue"}, subset=slice_first_cell)
    s.set_properties(**{"background-color": "lightred"}, subset=slice_first_col)
    # s.set_properties(**{"background-color": "lightyellow"}, subset=slice_rest)
    s.format_index(
        formatter=format_percentages, escape="latex", axis=1, level=1
    ).format_index(escape="latex", formatter=format_percentages, axis=0, level=1)
    s.format(precision=2)
    # Convert styled DataFrame to LaTeX table
    latex_table = s.to_latex(convert_css=True, hrules=True)
    # Print or save the LaTeX table
    print(latex_table)
