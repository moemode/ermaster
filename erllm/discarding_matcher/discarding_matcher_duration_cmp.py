"""
Calculates speedup factor of discarding matcher over LLM matcher.
"""

import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH
from erllm.utils import rename_datasets

SIM_FUNCTION = "overlap"

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(EVAL_FOLDER_PATH / "discarding_matcher_perf.csv")

    # Filter the data based on the specified similarity function
    df = df[df["Sim Function"] == SIM_FUNCTION]

    # Group the data by "Dataset" and get the first Sim Duration and LLM Matcher Duration
    # These values are constant for a given dataset, so we only need the first occurrence
    result = (
        df.groupby("Dataset")
        .agg({"Sim Duration": "first", "LLM Matcher Duration": "first"})
        .reset_index()
    )

    # Calculate the factor between LLM Matcher Duration and Sim Duration
    result["Factor"] = result["LLM Matcher Duration"] / result["Sim Duration"]

    # Rename the datasets
    result = rename_datasets(result, preserve_sampled=False)

    # Save the result to a new CSV file
    result.to_csv(
        EVAL_WRITEUP_FOLDER_PATH / "discarding_matcher_duration_cmp.csv", index=False
    )
