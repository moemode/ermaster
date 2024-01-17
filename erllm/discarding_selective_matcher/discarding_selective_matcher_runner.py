from pathlib import Path
from typing import Iterable, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import itertools
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import find_matching_csv
from erllm.discarding_selective_matcher.discarding_selective_matcher import (
    eval_discarding_selective_matcher,
)

CONFIGURATIONS = {
    "basic-cmp": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "label_fractions": [0, 0.05, 0.1, 0.15, 0.2],
        "discard_fractions": [0, 0.5, 0.7, 0.8],
        "outfolder": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "basic_cmp",
    },
    "grid": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "label_fractions": np.arange(0.0, 0.25 + 0.01, 0.01),
        "discard_fractions": np.arange(0.0, 1 + 0.01, 0.01),
        "outfolder": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "grid",
    },
}


def discarding_selective_matcher_runner(
    runfiles: Iterable[Path],
    similarity_files: Iterable[Path],
    label_fractions: Iterable[float],
    discard_fractions: Iterable[float],
    outfile: Optional[Path] = None,
):
    results = []
    for path in runfiles:
        params = list(itertools.product(label_fractions, discard_fractions))
        for label_fraction, discard_fraction in tqdm(params):
            if label_fraction + discard_fraction > 1:
                continue
            dataset_name = path.stem.split("-")[0]
            simPath = find_matching_csv(path, similarity_files)
            if not simPath:
                raise ValueError(f"No matching similarity file found for {path}")

            r = eval_discarding_selective_matcher(
                discard_fraction, label_fraction, path, simPath
            )
            r["Dataset"] = dataset_name
            r["Label Fraction"] = label_fraction
            r["Discard Fraction"] = discard_fraction
            results.append(r)
    df = pd.DataFrame(results)
    if outfile:
        df.to_csv(outfile, index=False)
    return df


if __name__ == "__main__":
    cfg_name = "basic-cmp"
    cfg = CONFIGURATIONS[cfg_name]
    cfg["outfolder"].mkdir(parents=True, exist_ok=True)
    result = discarding_selective_matcher_runner(
        list(cfg["runfiles"].glob("*.json")),
        list(Path(cfg["similarities"]).glob("*-allsim.csv")),
        cfg["label_fractions"],
        cfg["discard_fractions"],
    )
    result.to_csv(cfg["outfolder"] / "result.csv", index=False)
    # Group by Label Fraction, Discard Fraction, and Method
    grouped_result = result.groupby(["Label Fraction", "Discard Fraction"])
    # Calculate mean F1 for each group
    mean_f1 = grouped_result["F1"].mean().reset_index()
    mean_f1.to_csv(
        cfg["outfolder"] / "mean_f1.csv",
        index=False,
    )
