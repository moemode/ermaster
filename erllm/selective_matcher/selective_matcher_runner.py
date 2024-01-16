"""
This script runs the discarding matcher algorithm on multiple datasets with different threshold values.
It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration.
The results are stored in a pandas DataFrame and saved as a CSV file.
"""

from pathlib import Path
from typing import Iterable, Optional
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
        "param_range": np.arange(0.0, 0.20 + 0.01, 0.01),
        "random_tries": 30,
    },
    "base-selected": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "35_base_selected.csv",
        "param_range": [0.05, 0.1, 0.15],
        "random_tries": 30,
    },
    "gpt-4-base-cov": {
        "runfiles": RUNS_FOLDER_PATH / "4_base",
        "outpath": EVAL_FOLDER_PATH / "selective_matcher" / "4_base_covs.csv",
        "selective_matcher": selective_classifier_cov,
        "param_range": np.arange(0.0, 1 + 0.01, 0.01),
    },
}


def selective_matcher_runner(
    runfiles: Iterable[Path],
    label_fractions: Iterable[float],
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    results = []
    for path in runfiles:
        for label_fraction in label_fractions:
            dataset_name = path.stem.split("-")[0]
            truths, predictions, _, probabilities, _ = read_run(path)
            k = int(round(label_fraction * len(truths)))
            r = eval_selective_matcher(truths, predictions, probabilities, k)
            r["Dataset"] = dataset_name
            r["Label Fraction"] = label_fraction
            r["N_labeled"] = k
            results.append(r)
    df = pd.DataFrame(results)
    if outfile:
        df.to_csv(outfile, index=False)
    return df


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
            r_result["Fraction"] = f
            r_result["N_labeled"] = k
            sm_result["Dataset"] = dataset_name
            sm_result["Method"] = "SM"
            sm_result["Fraction"] = f
            sm_result["N_labeled"] = k
            dataset_results.append(r_result)
            dataset_results.append(sm_result)
        results.extend(dataset_results)
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(cfg["outpath"], index=False)
