"""
Generate and analyze performance/cost trade-off for the discarding matcher based on F1 decrease thresholds.
Calculates F1 decrease, relative cost, and relative duration for each dataset and threshold.
"""

from erllm import (
    EVAL_FOLDER_PATH,
    EVAL_WRITEUP_FOLDER_PATH,
    RUNS_FOLDER_PATH,
    SIMILARITIES_FOLDER_PATH,
)
from erllm.llm_matcher.cost import str_cost
from erllm.llm_matcher.evalrun import read_run_deprecated, read_run_raw
import pandas as pd

from erllm.utils import rename_datasets

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "sim_function": "overlap",
    },
}


def get_cost_time_abs(df: pd.DataFrame) -> None:
    """
    Save the DataFrame to a LaTeX table.
    """
    df = rename_datasets(df, preserve_sampled=False)
    df["Cost"] = df["Relative_Cost"] * df["LLM Matcher Cost"]
    df["Duration"] = df["Relative_Duration"] * df["LLM Matcher Duration"]
    df["F1_Decrease_Threshold"] = (df["F1_Decrease_Threshold"] * 100).round(2).abs()
    time_abs = df.pivot_table(
        index="Dataset",
        columns="F1_Decrease_Threshold",
        values="Duration",
    )
    # Add an extra column to time_abs for LLM Matcher Duration per dataset
    time_abs["LLM_Matcher_Duration"] = df.groupby("Dataset")[
        "LLM Matcher Duration"
    ].first()
    # make the LLM_Matcher_Duration column the first column
    time_abs = time_abs[
        ["LLM_Matcher_Duration"]
        + [col for col in time_abs if col != "LLM_Matcher_Duration"]
    ]

    # add extra column to time_abs for the LLM Matcher Duration per dataset
    cost_abs = df.pivot_table(
        index="Dataset", columns="F1_Decrease_Threshold", values="Cost"
    )
    cost_abs["LLM_Matcher_Cost"] = df.groupby("Dataset")["LLM Matcher Cost"].first()
    # make the LLM_Matcher_Cost column the first column
    cost_abs = cost_abs[
        ["LLM_Matcher_Cost"] + [col for col in cost_abs if col != "LLM_Matcher_Cost"]
    ]
    # order cost_abs decreasing by
    return time_abs, cost_abs


def to_latex_table(df: pd.DataFrame, caption: str, label: str, fname: str):
    s = df.style
    s.format(precision=2)
    latex_table = s.to_latex(
        EVAL_FOLDER_PATH / "discarding_matcher" / fname,
        # column_format="lccc",
        hrules=True,
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        caption=caption,
        label=label,
    )


if __name__ == "__main__":
    """
    Add the total duration and cost of running the LLM matcher to the discarding matcher tradeoff table.
    """
    cfg = CONFIGURATIONS["base"]
    df_tradeoff_rel = pd.read_csv(
        EVAL_WRITEUP_FOLDER_PATH / "discarding_matcher_tradeoff.csv"
    )
    # add columns LLM Matcher Cost and LLM Matcher Duration
    df_tradeoff_rel["LLM Matcher Cost"] = 0
    df_tradeoff_rel["LLM Matcher Duration"] = 0
    for path in cfg["runfiles"].glob("*force-gpt*.json"):
        dataset_name = path.stem.split("-")[0]
        truths, predictions, _, _, pair_ids = read_run_deprecated(path)
        completions = read_run_raw(path)
        prompts = list(map(lambda cp: cp.prompt_string, completions.values()))
        # cost of running basic matcher without discarder
        cost_llm_matcher = str_cost(prompts, 1, "gpt-3.5-turbo-instruct")
        duration_llm_matcher = sum(map(lambda cp: cp.duration, completions.values()))
        # in df_tradeoff_rel, replace LLM Matcher Cost and LLM Matcher Duration with the values calculated above
        df_tradeoff_rel.loc[
            df_tradeoff_rel["Dataset"] == dataset_name, "LLM Matcher Cost"
        ] = cost_llm_matcher
        df_tradeoff_rel.loc[
            df_tradeoff_rel["Dataset"] == dataset_name, "LLM Matcher Duration"
        ] = duration_llm_matcher
        # print(dataset_name, cost_llm_matcher, duration_llm_matcher)
    time_abs, cost_abs = get_cost_time_abs(df_tradeoff_rel)
    to_latex_table(
        time_abs,
        "Time for LLM Matcher and Discarding Matcher at various F1 Reduction Thresholds",
        "tab:f1_time_tradeoff_abs",
        "f1_time_tradeoff_abs.ltx",
    )
    to_latex_table(
        cost_abs,
        "Cost for LLM Matcher and Discarding Matcher at various F1 Reduction Thresholds",
        "tab:f1_cost_tradeoff_abs",
        "f1_cost_tradeoff_abs.ltx",
    )
