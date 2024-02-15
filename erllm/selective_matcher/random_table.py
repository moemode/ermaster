"""
Create a latex comparison table of F1 scores between LLM matcher and random labeling at different label fractions.
"""

import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH
from erllm.utils import rename_datasets


PLOT_METRICS = {
    "all": ["precision", "recall", "f1"],
    "f1": ["f1"],
}

CONFIGURATIONS = {
    "base": {
        "inpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": EVAL_FOLDER_PATH / "selective_matcher" / "35_base_rnd.ltx",
        "fractions": [0, 0.05, 0.1, 0.15],
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
    cfg = CONFIGURATIONS["base"]
    fractions = cfg["fractions"]
    df = pd.read_csv(cfg["inpath"])
    df = df[df["Method"] == "Random"]
    df = df[df["Fraction"].isin(fractions)]
    df = rename_datasets(df, preserve_sampled=False)
    # keep Dataset, F1, Fraction
    df = df[["Dataset", "Fraction", "F1"]]

    # Calculate mean F1 for each fraction
    mean_df = df.groupby("Fraction")["F1"].mean().reset_index()

    # Add "All" dataset for each fraction with mean F1
    all_datasets = pd.DataFrame(
        {"Dataset": ["All"] * len(fractions), "Fraction": fractions}
    )
    all_datasets["F1"] = mean_df["F1"].values

    # Concatenate the original DataFrame with the new "All" dataset entries
    df = pd.concat([df, all_datasets])

    df["F1 Absolute Increase"] = df.groupby("Dataset")["F1"].transform(
        lambda x: x - x.iloc[0]
    )
    df["F1 Relative Increase"] = df.groupby("Dataset")["F1"].transform(
        lambda x: (x - x.iloc[0]) / x.iloc[0] * 100
    )
    print(df)
    pivot_df = df.pivot(
        index="Dataset",
        columns="Fraction",
        values=["F1", "F1 Relative Increase"],
    )
    for fraction in fractions:
        if fraction == 0:
            pivot_df[("F1 & Relative Increase", fraction)] = (
                round(pivot_df[("F1", fraction)], 2).map("{:.2f}".format).astype(str)
            )
        else:
            pivot_df[("F1 & Relative Increase", fraction)] = (
                round(pivot_df[("F1", fraction)], 2).map("{:.2f}".format).astype(str)
                + " ("
                + round(pivot_df[("F1 Relative Increase", fraction)], 2)
                .map("{:+06.2f}%".format)
                .astype(str)
                + ")"
            )
    pivot_df = pivot_df.sort_values(by=("F1", 0.00))
    table_df = pivot_df["F1 & Relative Increase"]
    table_df.reset_index(inplace=True)
    table_df.columns = table_df.columns.map(
        {
            "Dataset": "Dataset",
            0.0: "LLM Matcher",
            0.05: "Random (5%)",
            0.1: "Random (10%)",
            0.15: "Random (15%)",
        }
    )
    print(table_df)
    table_df.to_latex(cfg["save_to"], escape=True, index=False)
