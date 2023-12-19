"""
Performs calibration analysis on language model predictions for different datasets, calculating Brier Score and Expected Calibration Error (ECE). 
It generates visualizations of reliability diagrams and saves the calibration metrics in CSV files, organized by model configurations
"""
from typing import Dict
from erllm import EVAL_FOLDER_PATH, FIGURE_FOLDER_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import read_run
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from erllm.calibration.reliability_diagrams import (
    reliability_diagrams,
    compute_calibration,
)
import numpy as np
import pandas as pd


def setup_plt():
    """
    Override matplotlib default styling.
    """
    plt.style.use("seaborn-v0_8")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def calibration_data(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Brier Score and Expected Calibration Error (ECE).

    Parameters:
        truths (numpy.ndarray): Ground truth labels.
        predictions (numpy.ndarray): Predicted labels.
        probabilities (numpy.ndarray): Predicted probabilities.

    Returns:
        dict: Dictionary containing Brier Score and ECE.
    """
    probabilities_brier = probabilities.copy()
    pred0 = 0 == predictions
    probabilities_brier[pred0] = 1 - probabilities_brier[pred0]
    brier = brier_score_loss(truths, probabilities_brier)
    ece = compute_calibration(truths, predictions, probabilities, num_bins=10)[
        "expected_calibration_error"
    ]
    probs_for_truth = np.where(predictions == truths, probabilities, 1 - probabilities)
    average_calibration_error = np.mean(1 - probs_for_truth)
    return {"Brier Score": brier, "ECE": ece, "ACE": average_calibration_error}


CONFIGURATIONS = {
    "3.5-hash": {
        "paths": (RUNS_FOLDER_PATH / "35_hash").glob("*.json"),
        "outpath_prefix": "hash",
    },
    "3.5-base": {
        "paths": (RUNS_FOLDER_PATH / "35_base").glob("*.json"),
        "outpath_prefix": "base",
    },
}

CALIBRATION_PATH = EVAL_FOLDER_PATH / "calibration"
CALIBRATION_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for config_name, config in CONFIGURATIONS.items():
        inpaths, prefix = config["paths"], config["outpath_prefix"]
        dfdata = []
        results = dict()

        for path in inpaths:
            truths, predictions, _, probabilities, _ = read_run(path)
            dataset_name = (
                path.stem.split("-")[0]
                .replace("structured_", "")
                .replace("textual_", "")
            )
            calibration_results = calibration_data(truths, predictions, probabilities)
            dfdata.append({"Dataset": dataset_name, **calibration_results})
            results[dataset_name] = {
                "true_labels": truths,
                "pred_labels": predictions,
                "confidences": probabilities,
            }

        # Create a DataFrame from the results list
        df = pd.DataFrame(dfdata)
        df.to_csv(
            EVAL_FOLDER_PATH / "calibration" / f"{prefix}_calibration.csv", index=False
        )

        # Display the DataFrame
        print(df)

        setup_plt()
        CALIBRATION_FIGURE_PATH = FIGURE_FOLDER_PATH / "calibration"
        CALIBRATION_FIGURE_PATH.mkdir(parents=True, exist_ok=True)
        fig = reliability_diagrams(
            results,
            num_bins=10,
            draw_bin_importance="alpha",
            num_cols=3,
            dpi=100,
            return_fig=True,
        )
        fig.savefig(
            CALIBRATION_FIGURE_PATH / f"{prefix}-calibration-all.png",
            format="png",
            dpi=144,
            bbox_inches="tight",
            pad_inches=0.2,
        )
