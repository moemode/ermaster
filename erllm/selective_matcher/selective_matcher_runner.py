"""
This script runs the discarding matcher algorithm on multiple datasets with different threshold values.
It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration.
The results are stored in a pandas DataFrame and saved as a CSV file.
"""

import numpy as np
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.selective_matcher.selective_matcher import selective_matcher

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base.csv",
    },
}

if __name__ == "__main__":
    cfg = CONFIGURATIONS["base"]
    cfg["outpath"].parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Create values in the range 0.0 to 1.0 with an increment of 0.05
    inc = 0.01
    threshold_values = np.arange(0.0, 1 + inc, inc)

    for path in cfg["runfiles"].glob("*force-gpt*.json"):
        dataset_name = path.stem.split("-")[0]
        # Call discarding_matcher and store the results in the dataframe
        dataset_results = selective_matcher(path, threshold_values)
        # make into dataframe, add column with dataset name
        for r in dataset_results:
            r["Dataset"] = dataset_name
        print(dataset_results)
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv(cfg["outpath"], index=False)
