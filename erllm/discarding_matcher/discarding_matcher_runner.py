"""
Runs the discarding matcher algorithm on multiple datasets with different threshold values.
It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration.
"""

from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import (
    discarding_matcher_cov,
    find_matching_csv,
    discarding_matcher,
)


def discarding_matcher_cov_runner(
    runfiles: Iterable[Path],
    simfiles: Iterable[Path],
    discard_fractions: Iterable[float],
    sim_function="overlap",
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    results = []
    for path in runfiles:
        dataset_name = path.stem.split("-")[0]
        simpath = find_matching_csv(path, simfiles)
        if not simpath:
            raise ValueError(f"No matching similarity file found for {path}")
        for f in discard_fractions:
            r = discarding_matcher_cov(f, path, simpath, sim_function)
            r["Dataset"] = dataset_name
            r["Discard Fraction"] = f
            results.append(r)
    df = pd.DataFrame(results)
    if outfile:
        df.to_csv(outfile, index=False)
    return df


CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "sim_function": "overlap",
    },
}


if __name__ == "__main__":
    results = []
    # Create values in the range 0.0 to 1.0 with an increment of 0.05
    inc = 0.01
    threshold_values = np.arange(0.0, 1 + inc, inc)
    cfg = CONFIGURATIONS["base"]
    for threshold in threshold_values:
        for path in cfg["runfiles"].glob("*force-gpt*.json"):
            simPath = find_matching_csv(
                path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
            )
            if not simPath:
                raise ValueError(
                    f"No matching similarity file in {cfg['similarities']} found for {path}"
                )
            results.append(
                discarding_matcher(threshold, path, simPath, cfg["sim_function"])
            )
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv(EVAL_FOLDER_PATH / "discarding_matcher_perf.csv", index=False)
