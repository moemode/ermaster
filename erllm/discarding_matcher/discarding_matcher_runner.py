"""
This script runs the discarding matcher algorithm on multiple datasets with different threshold values.
It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration.
The results are stored in a pandas DataFrame and saved as a CSV file.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import (
    find_matching_csv,
    discarding_matcher,
)

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
    },
}

if __name__ == "__main__":
    results = []
    # Create values in the range 0.0 to 1.0 with an increment of 0.05
    inc = 0.01
    threshold_values = np.arange(0.0, 1 + inc, inc)
    for threshold in threshold_values:
        for path in CONFIGURATIONS["base"]["runfiles"].glob("*force-gpt*.json"):
            dataset_name = path.stem.split("-")[0]
            simPath = find_matching_csv(
                path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
            )
            if not simPath:
                raise ValueError(
                    f"No matching similarity file in {CONFIGURATIONS['base']['similarities']} found for {path}"
                )
            # Call discarding_matcher and store the results in the dataframe
            (
                acc,
                prec,
                rec,
                f1,
                cost,
                cost_rel,
                duration,
                duration_rel,
            ) = discarding_matcher(threshold, path, simPath)
            results.append(
                {
                    "Dataset": dataset_name,
                    "Threshold": threshold,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "Cost": cost,
                    "Cost Relative": cost_rel,
                    "Duration": duration,
                    "Duration Relative": duration_rel,
                }
            )
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv(EVAL_FOLDER_PATH / "discarding_matcher_perf.csv", index=False)
