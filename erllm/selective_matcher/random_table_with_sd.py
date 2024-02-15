"""
Creates a latex table displaying std. dev. of F1 scores for different fractions of random labeling.
"""

import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH
from erllm.utils import rename_datasets


CONFIGURATIONS = {
    "base": {
        "inpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base.csv",
        "save_to": EVAL_FOLDER_PATH / "selective_matcher" / "35_base_sd.ltx",
        "agg_save_to": EVAL_FOLDER_PATH / "selective_matcher" / "35_base_sd_agg.ltx",
        "fractions": [0, 0.05, 0.1, 0.15],
    },
    "base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base_covs.csv",
        "save_to": EVAL_WRITEUP_FOLDER_PATH
        / "selective_classifier_tradeoff_35_base.ltx",
    },
    "gpt-4-base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "4_base_covs.csv",
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
    df = df[["Dataset", "Fraction", "F1 SD"]]
    """
    # Calculate mean F1 for each fraction
    mean_df = df.groupby("Fraction")["F1"].mean().reset_index()

    # Add "All" dataset for each fraction with mean F1
    all_datasets = pd.DataFrame(
        {"Dataset": ["All"] * len(fractions), "Fraction": fractions}
    )
    all_datasets["F1"] = mean_df["F1"].values

    # Concatenate the original DataFrame with the new "All" dataset entries
    df = pd.concat([df, all_datasets])
    """
    # Format columns as required
    df = df[df["Fraction"].isin([0.05, 0.10, 0.15])]
    # Find the largest and smallest F1 SD for each fraction
    max_std = df.groupby("Fraction")["F1 SD"].max().reset_index()
    min_std = df.groupby("Fraction")["F1 SD"].min().reset_index()

    # Merge the DataFrames to get both the largest and smallest F1 SD for each fraction
    std_info_df = pd.merge(max_std, min_std, on="Fraction", suffixes=("_max", "_min"))

    # Rename columns for clarity
    std_info_df = std_info_df.rename(
        columns={"F1 SD_max": "Max F1 SD", "F1 SD_min": "Min F1 SD"}
    )
    std_info_df["SD Range"] = std_info_df.apply(
        lambda row: f'{row["Min F1 SD"]:.4f} - {row["Max F1 SD"]:.4f}', axis=1
    )
    std_info_df = std_info_df[["Fraction", "SD Range"]]
    std_info_df.to_latex(cfg["agg_save_to"], escape=True, index=False)

    # Display the new DataFrame with information about the largest and smallest F1 SD for each fraction
    print(std_info_df)
    df["Fraction"] = df["Fraction"].apply(lambda x: f"{x:.2f}")
    df["F1 SD"] = df["F1 SD"].apply(lambda x: f"{x:.4f}")
    # only keep rows with fraction 0.0 and 0.15
    df.to_latex(cfg["save_to"], escape=True, index=False)
