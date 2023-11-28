from evalrun import read_run_alternate
from sklearn.metrics import brier_score_loss
from pathlib import Path
import matplotlib.pyplot as plt
from reliability_diagrams import *
import pandas as pd


def setup_plt():
    # Override matplotlib default styling.
    plt.style.use("seaborn-v0_8")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def calibration_data(truths, predictions, probabilities):
    probabilities_brier = probabilities.copy()
    pred0 = 0 == predictions
    probabilities_brier[pred0] = 1 - probabilities_brier[pred0]
    brier = brier_score_loss(truths, probabilities_brier)
    ece = compute_calibration(truths, predictions, probabilities, num_bins=10)[
        "expected_calibration_error"
    ]
    return {"Brier Score": brier, "ECE": ece}


CONFIGURATIONS = {
    "3.5-hash": {
        "paths": Path("/home/v/coding/ermaster/runs/35_hash").glob("*.json"),
        "outpath_prefix": "hash",
    },
    "3.5-base": {
        "paths": Path("/home/v/coding/ermaster/runs/35_base").glob("*.json"),
        "outpath_prefix": "base",
    },
}

if __name__ == "__main__":
    for config_name, config in CONFIGURATIONS.items():
        inpaths, prefix = config["paths"], config["outpath_prefix"]
        dfdata = []
        results = dict()

        for path in inpaths:
            truths, predictions, _, probabilities, _ = read_run_alternate(path)
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
        df.to_csv(f"eval_writeup/{prefix}_calibration.csv", index=False)

        # Display the DataFrame
        print(df)

        setup_plt()
        fig = reliability_diagrams(
            results,
            num_bins=10,
            draw_bin_importance="alpha",
            num_cols=3,
            dpi=100,
            return_fig=True,
        )
        fig.savefig(
            f"figures/{prefix}-calibration-all.png",
            format="png",
            dpi=144,
            bbox_inches="tight",
            pad_inches=0.2,
        )
