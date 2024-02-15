"""
Runs selective classification over ranges of threshold/coverage parameters on multiple datasets.
"""

import numpy as np
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.selective_classifier.selective_classifier import (
    selective_classifier,
    selective_classifier_cov,
)

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "outpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base.csv",
        "selective_classifier": selective_classifier,
        "param_range": np.arange(0.500, 1 + 0.01, 0.01),
    },
    "base-cov": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "outpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base_covs.csv",
        "selective_classifier": selective_classifier_cov,
        "param_range": np.arange(0.0, 1 + 0.01, 0.01),
    },
    "gpt-4-base-cov": {
        "runfiles": RUNS_FOLDER_PATH / "4_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "outpath": EVAL_FOLDER_PATH / "selective_classifier" / "4_base_covs.csv",
        "selective_classifier": selective_classifier_cov,
        "param_range": np.arange(0.0, 1 + 0.01, 0.01),
    },
}

if __name__ == "__main__":
    cfg = CONFIGURATIONS["gpt-4-base-cov"]
    cfg["outpath"].parent.mkdir(parents=True, exist_ok=True)
    results = []
    for path in cfg["runfiles"].glob("*force-gpt*.json"):
        dataset_name = path.stem.split("-")[0]
        # Call discarding_matcher and store the results in the dataframe
        dataset_results = cfg["selective_classifier"](path, cfg["param_range"])
        # make into dataframe, add column with dataset name
        for r in dataset_results:
            r["Dataset"] = dataset_name
        results.extend(dataset_results)
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv(cfg["outpath"], index=False)
