"""
Creates a table for comparing different matcher architectures based on their discarding error, cost, time
and classification metrics.
"""

from pathlib import Path
from typing import Iterable
import pandas as pd
from pandas.io.formats.style import Styler
from erllm import EVAL_FOLDER_PATH

CONFIGURATIONS = {
    "recommended": {
        "result_folder": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "recommended",
        "order": [(0.0, 0.0), (0.8, 0.0), (0.0, 0.15), (0.8, 0.15), (0.5, 0.15)],
    },
}


def format_percentages(c: float, decimals: int = 0) -> str:
    """
    Formats a given float value as a percentage string.

    Args:
        c (float): The float value to be formatted as a percentage.
        decimals (int, optional): The number of decimal places to include in the formatted percentage. Defaults to 0.

    Returns:
        str: The formatted percentage string.
    """
    return f"{c*100:.{decimals}f}%"


COLUMN_SHORTHANDS = {
    "Discard Fraction": "Discard",
    "Label Fraction": "Label",
    "Discarding Error": "Disc. Error",
    "Duration": "Time (s)",
    "LLM Cost": "LLM Cost (\$)",
    "Recall": "Rec.",
    "Precision": "Prec.",
}


def prepare_data(
    df: pd.DataFrame, order: Iterable[tuple[float, float]], save_to: Path
) -> pd.DataFrame:
    """
    Prepare data for analysis by computing changes relative to the LLM matcher and save the result to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        order (Iterable[tuple[float, float]]): The order of the data.
        save_to (Path): The path to save the result CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame with changes relative to the LLM matcher (no discard, no label).

    """
    df["Discarding Error"] = df["Discarded FN"] / df["Discarded"]
    df["Duration"] = df["Discarder Duration"] + df["LLM Duration"]
    df["Speedup"] = 1 - df["Duration"] / df["LLM All Duration"]
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
                "Config ID": "first",
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
    # order rows by Config ID
    result_df = result_df.sort_values("Config ID")
    # Save the result to a CSV file
    result_df.to_csv(save_to / "allstats.csv", index=False)
    return result_df


def build_table(df: pd.DataFrame, save_to: Path) -> None:
    """
    Create a table for comparing different matcher architectures based on their discarding error, cost, time
    and classification metrics. The table is styled and saved as a LaTeX table.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        save_to (Path): The path where the table should be saved.

    Returns:
        None
    """
    table_df = df[
        [
            "Discard Fraction",
            "Label Fraction",
            "Discarding Error",
            "LLM Cost",
            "Duration",
            "Recall",
            "Precision",
            "F1",
        ]
    ]

    # apply format_percentages to Discard Fraction and Label Fraction
    table_df["Discard Fraction"] = table_df["Discard Fraction"].apply(
        format_percentages
    )
    table_df["Label Fraction"] = table_df["Label Fraction"].apply(format_percentages)
    table_df["Discarding Error"] = table_df["Discarding Error"].apply(
        format_percentages, decimals=2
    )
    # convert Discarding Error to string map nan to '-'
    table_df["Discarding Error"] = (
        table_df["Discarding Error"].astype(str).replace("nan\%", "-")
    )
    # add new column as first column
    table_df.insert(0, "Purpose", "Example Purpose")
    # Round LLM Cost to 2 decimal places
    table_df["LLM Cost"] = table_df["LLM Cost"].round(2)
    # Round Duration to 0 decimal places as int
    table_df["Duration"] = table_df["Duration"].round(0).astype(int)
    # Round Recall, Precision, and F1 to 2 decimal places
    table_df["Recall"] = table_df["Recall"].round(2)
    table_df["Precision"] = table_df["Precision"].round(2)
    table_df["F1"] = table_df["F1"].round(2)
    # map column names to shorthands
    table_df = table_df.rename(columns=COLUMN_SHORTHANDS)
    s = highlight_cells(table_df)
    s.format(precision=2)
    s.hide()
    ltable = s.to_latex(
        save_to / f"allstats_table.tex",
        convert_css=True,
        hrules=True,
        position_float="centering",
        multicol_align="c",
        caption=f"Test",
    )


def highlight_cells(table_df) -> Styler:
    """
    Highlights specific cells in a DataFrame based on certain conditions.
    Preparation for converting to latex table.

    Args:
        table_df (DataFrame): The input DataFrame.

    Returns:
        Styler: The styled DataFrame.

    """
    cname_label_fraction = COLUMN_SHORTHANDS["Label Fraction"]
    cname_discard_error = COLUMN_SHORTHANDS["Discarding Error"]
    cname_llm_cost = COLUMN_SHORTHANDS["LLM Cost"]
    cname_duration = COLUMN_SHORTHANDS["Duration"]

    s = table_df.style
    df_label_fraction = table_df[table_df[cname_label_fraction] == "0\%"]
    green_label_fraction = pd.IndexSlice[df_label_fraction.index, cname_label_fraction]
    yellow_label_fraction = pd.IndexSlice[
        table_df.index.difference(df_label_fraction.index), cname_label_fraction
    ]
    s.set_properties(**{"background-color": "lightgreen"}, subset=green_label_fraction)
    s.set_properties(
        **{"background-color": "lightyellow"}, subset=yellow_label_fraction
    )

    df_discard_error = table_df[table_df[cname_discard_error] == "-"]
    green_discard_error = pd.IndexSlice[df_discard_error.index, cname_discard_error]
    yellow_discard_error = pd.IndexSlice[
        table_df.index.difference(df_discard_error.index), cname_discard_error
    ]
    s.set_properties(**{"background-color": "lightgreen"}, subset=green_discard_error)
    s.set_properties(**{"background-color": "lightyellow"}, subset=yellow_discard_error)
    # color llm cost and duration
    max_llm_cost_value = table_df[cname_llm_cost].max()
    df_llm_cost = table_df[table_df[cname_llm_cost] != max_llm_cost_value]
    green_llm_cost = pd.IndexSlice[df_llm_cost.index, [cname_llm_cost, cname_duration]]
    yellow_llm_cost = pd.IndexSlice[
        table_df.index.difference(df_llm_cost.index), [cname_llm_cost, cname_duration]
    ]
    s.set_properties(**{"background-color": "lightgreen"}, subset=green_llm_cost)
    s.set_properties(**{"background-color": "lightyellow"}, subset=yellow_llm_cost)
    return s


if __name__ == "__main__":
    cfg_name = "recommended"
    cfg = CONFIGURATIONS[cfg_name]
    df = pd.read_csv(cfg["result_folder"] / "result.csv")
    tdf = prepare_data(df, cfg["order"], cfg["result_folder"])
    build_table(tdf, cfg["result_folder"])
