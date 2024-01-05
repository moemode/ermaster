"""
This script runs the discarding matcher algorithm on multiple datasets with different threshold values.
It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration.
The results are stored in a pandas DataFrame and saved as a CSV file.
"""

import numpy as np
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import read_run
from erllm.selective_classifier.selective_classifier import (
    selective_classifier_cov,
)
from erllm.selective_matcher.selective_matcher import (
    eval_label_k_random,
    eval_selective_matcher,
)

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base.csv",
        "param_range": np.arange(0.0, 0.15 + 0.01, 0.01),
        "random_tries": 30,
    },
    "base-cov": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base_covs.csv",
        "selective_matcher": selective_classifier_cov,
        "param_range": np.arange(0.0, 1 + 0.01, 0.01),
    },
    "gpt-4-base-cov": {
        "runfiles": RUNS_FOLDER_PATH / "4_base",
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "4_base_covs.csv",
        "selective_matcher": selective_classifier_cov,
        "param_range": np.arange(0.0, 1 + 0.01, 0.01),
    },
}

if __name__ == "__main__":
    cfg = CONFIGURATIONS["base"]
    cfg["outpath"].parent.mkdir(parents=True, exist_ok=True)
    results = []
    for path in cfg["runfiles"].glob("*force-gpt*.json"):
        dataset_results = []
        dataset_name = path.stem.split("-")[0]
        truths, predictions, _, probabilities, _ = read_run(path)
        for f in cfg["param_range"]:
            k = int(round(f * len(truths)))
            r_result = eval_label_k_random(truths, predictions, k, cfg["random_tries"])
            sm_result = eval_selective_matcher(truths, predictions, probabilities, k)
            r_result["Dataset"] = dataset_name
            r_result["Method"] = "Random"
            sm_result["Dataset"] = dataset_name
            sm_result["Method"] = "SM"
            dataset_results.append(r_result)
            dataset_results.append(sm_result)
        results.extend(dataset_results)
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv(cfg["outpath"], index=False)
