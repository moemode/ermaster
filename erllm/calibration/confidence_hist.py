import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from erllm import FIGURE_FOLDER_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import read_run
from erllm.utils import my_setup_plt


CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "outfolder": FIGURE_FOLDER_PATH / "calibration" / "confidence_hist" / "base",
    },
}


def hist_plot(df: pd.DataFrame, save_to: Path):
    g = sns.histplot(
        data=df,
        x="probabilities",
        hue="outcome",
        multiple="stack",
        bins=10,
        common_norm=False,
    )
    """
    plt.title(f"Confidence Outcome Histogram for {cfg_name} Configuration")
    plt.xlabel("Probabilities")
    plt.ylabel("Frequency")
    plt.legend(title="Outcome")
    plt.savefig(cfg["outfolder"] / f"confidence_histogram_all_datasets.png")
    plt.show()
    """
    # Adjust layout
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle("Metrics vs Threshold for Different Datasets", fontsize=16)
    g.figure.savefig(save_to)


def confidence_outcome(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    result_df = pd.DataFrame()
    result_df["truths"] = truths
    result_df["predictions"] = predictions
    result_df["probabilities"] = probabilities
    # Determine outcome based on truth and prediction values
    result_df["outcome"] = np.select(
        [
            (truths == 1) & (predictions == 1),
            (truths == 0) & (predictions == 0),
            (truths == 0) & (predictions == 1),
            (truths == 1) & (predictions == 0),
        ],
        ["TP", "TN", "FP", "FN"],
    )
    return result_df


if __name__ == "__main__":
    my_setup_plt()
    cfg_name = "base"
    cfg = CONFIGURATIONS[cfg_name]
    cfg["outfolder"].mkdir(parents=True, exist_ok=True)
    confidence_outcome_df = pd.DataFrame()
    for path in cfg["runfiles"].glob("*.json"):
        dataset_name = path.stem.split("-")[0].replace("_1250", "")
        truths, predictions, _, confidences, _ = read_run(path)
        r = confidence_outcome(truths, predictions, confidences)
        # add r to confidence_outcome_df but with 'Dataset' column set to dataset_name
        r["Dataset"] = dataset_name
        confidence_outcome_df = pd.concat([confidence_outcome_df, r], ignore_index=True)
    combinations = [
        # ("TP",),
        # ("TN",),
        # ("FP",),
        # ("FN",),
        # ("TP", "FN"),
        # ("TN", "FP"),
        ("TP", "FP"),
        # ("TP", "TN", "FP", "FN"),
    ]
    # Create histograms for all datasets
    for c in combinations:
        df = confidence_outcome_df[confidence_outcome_df["outcome"].isin(c)]
        hist_plot(
            df,
            cfg["outfolder"] / f"confidence_histogram_all_datasets_{''.join(c)}.png",
        )

    """
    # Create histograms for each individual dataset
    datasets = confidence_outcome_df["Dataset"].unique()
    for dataset in datasets:
        plt.figure(figsize=(12, 6))
        subset_df = confidence_outcome_df[confidence_outcome_df["Dataset"] == dataset]
        sns.histplot(
            data=subset_df, x="probabilities", hue="outcome", multiple="stack", bins=20
        )
        plt.title(
            f"Confidence Outcome Histogram for {dataset} Dataset ({cfg_name} Configuration)"
        )
        plt.xlabel("Probabilities")
        plt.ylabel("Frequency")
        plt.legend(title="Outcome")
        plt.savefig(cfg["outfolder"] / f"confidence_histogram_{dataset}.png")
        plt.show()
    """
