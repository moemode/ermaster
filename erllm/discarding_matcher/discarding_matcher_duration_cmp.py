import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH
from erllm.utils import rename_datasets

SIM_FUNCTION = "overlap"

if __name__ == "__main__":
    df = pd.read_csv(EVAL_FOLDER_PATH / "discarding_matcher_perf.csv")
    # Filter to only include the overlap similarity function
    df = df[df["Sim Function"] == SIM_FUNCTION]
    # Group by "Dataset" and get first Sim Duration and LLM Matcher Duration
    # these are constant for a given dataset so just get first
    result = (
        df.groupby("Dataset")
        .agg({"Sim Duration": "first", "LLM Matcher Duration": "first"})
        .reset_index()
    )
    result["Factor"] = result["LLM Matcher Duration"] / result["Sim Duration"]
    result = rename_datasets(result, preserve_sampled=False)
    result.to_csv(
        EVAL_WRITEUP_FOLDER_PATH / "discarding_matcher_duration_cmp.csv", index=False
    )
