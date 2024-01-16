from pathlib import Path
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher_runner import (
    discarding_matcher_cov_runner,
)
from erllm.discarding_selective_matcher.discarding_selective_matcher_runner import (
    discarding_selective_matcher_runner,
)
from erllm.selective_matcher.selective_matcher_runner import selective_matcher_runner

CONFIGURATIONS = {
    "base": {
        "runfiles": RUNS_FOLDER_PATH / "35_base",
        "similarities": SIMILARITIES_FOLDER_PATH,
        "label_fractions": [0.05, 0.1, 0.15, 0.2],
        "discard_fractions": [0.5, 0.8, 0.9],
        "dsm_outfile": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "f1_comp_0"
        / "dsm.csv",
        "sm_outfile": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "f1_comp_0"
        / "sm.csv",
        "dm_outfile": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "f1_comp_0"
        / "dm.csv",
        "result_outfile": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "f1_comp_0"
        / "result.csv",
    },
}


if __name__ == "__main__":
    cfg_name = "base"
    cfg = CONFIGURATIONS[cfg_name]
    cfg["dsm_outfile"].parent.mkdir(parents=True, exist_ok=True)
    runfiles = list(cfg["runfiles"].glob("*.json"))
    simfiles = list(Path(cfg["similarities"]).glob("*-allsim.csv"))
    dsm_result = discarding_selective_matcher_runner(
        runfiles,
        simfiles,
        cfg["label_fractions"],
        cfg["discard_fractions"],
        cfg["dsm_outfile"],
    )
    dsm_result["Method"] = "DSM"
    sm_result = selective_matcher_runner(
        runfiles, cfg["label_fractions"], cfg["sm_outfile"]
    )
    sm_result["Method"] = "SM"
    dm_result = discarding_matcher_cov_runner(
        runfiles, simfiles, cfg["discard_fractions"], "overlap", cfg["dm_outfile"]
    )
    dm_result["Method"] = "DM"
    # concat result dataframes
    result = pd.concat([dsm_result, sm_result, dm_result])
    result.to_csv(cfg["result_outfile"], index=False)
