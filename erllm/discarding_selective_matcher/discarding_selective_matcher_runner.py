from pathlib import Path
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

if __name__ == "__main__":
    cfg_name = "base"
    cfg = CONFIGURATIONS[cfg_name]
    cfg["outfile"].parent.mkdir(parents=True, exist_ok=True)
    results = []
    for path in cfg["runfiles"].glob("*.json"):
        for label_fraction, discard_fraction in itertools.product(
            cfg["label_fractions"], cfg["discard_fractions"]
        ):
            if label_fraction + discard_fraction > 1:
                continue
            dataset_name = path.stem.split("-")[0]
            simPath = find_matching_csv(
                path, Path(cfg["similarities"]).glob("*-allsim.csv")
            )
            if not simPath:
                raise ValueError(
                    f"No matching similarity file in {cfg['similarities']} found for {path}"
                )

            r = eval_discarding_selective_matcher(
                discard_fraction, label_fraction, path, simPath
            )
            r["Dataset"] = dataset_name
            r["Label Fraction"] = label_fraction
            r["Discard Fraction"] = discard_fraction
            results.append(r)
    df = pd.DataFrame(results)
    df.to_csv(cfg["outfile"], index=False)
