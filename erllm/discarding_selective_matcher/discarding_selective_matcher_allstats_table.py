from pathlib import Path
import pandas as pd
from erllm import EVAL_FOLDER_PATH

CONFIGURATIONS = {
    "recommended": {
        "result_folder": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "recommended",
    },
}


def format_percentages(c):
    return f"{c*100:.0f}\%"


def build_table(df: pd.DataFrame, save_to: Path) -> str:
    df["Discarding Error"] = df["Discarded FN"] / df["Discarded"]
    df["Duration"] = df["Discarder Duration"] + df["LLM Duration"]
    df["Speedup"] = 1 - df["Duration"] / df["LLM All Duration"]
    # df = df.groupby(["Label Fraction", "Discard Fraction"])
    result_df = (
        df.groupby(["Discard Fraction", "Label Fraction"])
        .agg(
            {
                "Discarding Error": "mean",
                "Duration": "sum",
                "LLM Cost": "sum",
                "Recall": "mean",
                "Precision": "mean",
                "F1": "mean",
            }
        )
        .reset_index()
    )
    # Compute relative changes
    reference_row = result_df[
        (result_df["Label Fraction"] == 0) & (result_df["Discard Fraction"] == 0)
    ].iloc[0]
    result_df["Duration Change"] = (
        result_df["Duration"] - reference_row["Duration"]
    ) / reference_row["Duration"]
    result_df["LLM Cost Change"] = (
        result_df["LLM Cost"] - reference_row["LLM Cost"]
    ) / reference_row["LLM Cost"]
    result_df["Recall Change"] = (
        result_df["Recall"] - reference_row["Recall"]
    ) / reference_row["Recall"]
    result_df["Precision Change"] = (
        result_df["Precision"] - reference_row["Precision"]
    ) / reference_row["Precision"]
    result_df["F1 Change"] = (result_df["F1"] - reference_row["F1"]) / reference_row[
        "F1"
    ]

    # Save the result to a CSV file
    result_df.to_csv(save_to / "allstats.csv", index=False)
    return "hi"
    """
    # Calculate mean for each metric
    df = df[metric].mean().reset_index()
    pivot_df = df.pivot(
        index="Discard Fraction", columns="Label Fraction", values=metric
    )
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
    first_cell = idx[idx[*first_row], idx[*first_col]]
    first_row = idx[idx[*first_row], idx[second_col[0], second_col[1] :]]
    first_col = idx[idx[second_row[0], second_row[1] :], idx[*first_col]]
    slice_rest = idx[
        idx[second_row[0], second_row[1] :], idx[second_col[0], second_col[1] :]
    ]
    s = pivot_df.style
    s.set_properties(**{"background-color": "lightgreen"}, subset=first_row)
    s.set_properties(**{"background-color": "lightblue"}, subset=first_cell)
    s.set_properties(**{"background-color": "lightred"}, subset=first_col)
    # s.set_properties(**{"background-color": "lightyellow"}, subset=slice_rest)
    s.format_index(
        formatter=format_percentages, escape="latex", axis=1, level=1
    ).format_index(escape="latex", formatter=format_percentages, axis=0, level=1)
    s.format(precision=2)
    # Convert styled DataFrame to LaTeX table
    latex_table = s.to_latex(
        save_to / f"{metric.lower()}_table.tex",
        convert_css=True,
        hrules=True,
        position_float="centering",
        multicol_align="c",
        caption=f"{metric} scores for the discarding selective matcher with different label and discard fractions.",
    )
    # Print or save the LaTeX table
    return latex_table
    """


if __name__ == "__main__":
    cfg_name = "recommended"
    cfg = CONFIGURATIONS[cfg_name]
    df = pd.read_csv(cfg["result_folder"] / "result.csv")
    build_table(df, cfg["result_folder"])
