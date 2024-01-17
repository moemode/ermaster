from typing import Iterable
import pandas as pd
from pathlib import Path
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


def eval(result_folder: Path, mean_metrics: Iterable[str]):
    result_file = result_folder / "result.csv"
    result = pd.read_csv(result_file)
    # Group by Label Fraction, Discard Fraction, and Method
    grouped_result = result.groupby(["Label Fraction", "Discard Fraction"])
    for metric in mean_metrics:
        # Calculate mean for each metric
        mean_metric = grouped_result[metric].mean().reset_index()
        mean_metric.to_csv(
            result_folder / f"mean_{metric.lower()}.csv",
            index=False,
        )


if __name__ == "__main__":
    cfg_name = "grid"
    cfg = CONFIGURATIONS[cfg_name]
    eval(cfg["result_folder"], cfg["means"])
