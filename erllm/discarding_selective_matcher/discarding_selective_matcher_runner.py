from pathlib import Path
from typing import Iterable
import pandas as pd
import itertools
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import find_matching_csv
from erllm.discarding_selective_matcher.discarding_selective_matcher import (
    eval_discarding_selective_matcher,
)

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "label_fractions": [0.05, 0.1, 0.15, 0.2],
        "discard_fractions": [0.5, 0.8, 0.9],
        "outfile": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "perf.csv",
    },
}


def discarding_selective_matcher_runner(
    runfiles: Iterable[Path],
    similarity_files: Iterable[Path],
    label_fractions: Iterable[float],
    discard_fractions: Iterable[float],
    outfile: Path,
):
    results = []
    for path in runfiles:
        for label_fraction, discard_fraction in itertools.product(
            label_fractions, discard_fractions
        ):
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
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    cfg_name = "base"
    cfg = CONFIGURATIONS[cfg_name]
    cfg["outfile"].parent.mkdir(parents=True, exist_ok=True)
    discarding_selective_matcher_runner(
        list(cfg["runfiles"].glob("*.json")),
        list(Path(cfg["similarities"]).glob("*-allsim.csv")),
        cfg["label_fractions"],
        cfg["discard_fractions"],
        cfg["outfile"],
    )
