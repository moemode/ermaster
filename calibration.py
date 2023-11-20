from evalrun import read_run
from sklearn.metrics import brier_score_loss
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from reliability_diagrams import *


# Override matplotlib default styling.
# plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)

if __name__ == "__main__":
    results = dict()
    dfdata = []
    for path in Path("/home/v/coding/ermaster/runs").glob("*force_hash-gpt*.json"):
        truths, predictions, entropies, probabilities = read_run(path)
        correct = (truths == predictions).astype(int)
        # print(path.stem, brier_score_loss(correct, probabilities))
        dataset_name = (
            path.stem.split("-")[0].replace("structured_", "").replace("textual_", "")
        )
        brier = brier_score_loss(correct, probabilities)
        ece = compute_calibration(truths, predictions, probabilities, num_bins=10)[
            "expected_calibration_error"
        ]
        dfdata.append({"Dataset": dataset_name, "Brier Score": brier, "ECE": ece})

        results[dataset_name] = {
            "true_labels": truths,
            "pred_labels": predictions,
            "confidences": probabilities,
        }
        # Create a DataFrame from the results list
    df = pd.DataFrame(dfdata)

    # Display the DataFrame
    print(df)
    fig = reliability_diagrams(
        results,
        num_bins=10,
        draw_bin_importance="alpha",
        num_cols=3,
        dpi=100,
        return_fig=True,
    )
    fig.savefig(
        "figures/calibration-all.png",
        format="png",
        dpi=144,
        bbox_inches="tight",
        pad_inches=0.2,
    )
