"""
This script generates performance comparison plots for the discarding matcher.
It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots
for each dataset with different configurations, such as all metrics, no cost, and F1 with cost.
It also creates plots with showing the performance on all datasets at once.
"""
import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH, FIGURE_FOLDER_PATH
from erllm.utils import rename_datasets


PLOT_METRICS = {
    "all": ["precision", "recall", "f1"],
    "f1": ["f1"],
}

CONFIGURATIONS = {
    "base": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": FIGURE_FOLDER_PATH / "selective_classifier" / "base",
    },
    "base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base_covs.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": EVAL_WRITEUP_FOLDER_PATH
        / "selective_classifier_tradeoff_35_base.ltx",
    },
    "gpt-4-base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "4_base_covs.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": EVAL_WRITEUP_FOLDER_PATH
        / "selective_classifier_tradeoff_4_base.ltx",
    },
}


if __name__ == "__main__":
    cfg = CONFIGURATIONS["gpt-4-base-cov"]
    df = pd.read_csv(cfg["inpath"])
    # from 1 to 0.75 in steps of 0.05
    coverages = [1.0, 0.95, 0.9, 0.85, 0.8]
    df = df[df["coverage"].isin(coverages)]
    df = rename_datasets(df, preserve_sampled=False)
    # keep only Dataset, coverage, f1
    df = df[["Dataset", "coverage", "f1"]]
    print(df)
    # Pivot the DataFrame to create separate columns for each coverage
    df_pivot = df.pivot(index="Dataset", columns="coverage", values="f1").reset_index()
    # reorder columns from largest to smallest
    df_pivot = df_pivot[["Dataset", *coverages]]
    # Sort descendingly of values in colum 1.0
    df_pivot.sort_values(by=1.0, ascending=False, inplace=True)
    print(df_pivot)
    # convert all column names to string
    df_pivot.columns = df_pivot.columns.astype(str)
    df_pivot.to_latex(
        cfg["save_to"],
        index=False,
        escape=True,
        float_format="%.3f",
    )
