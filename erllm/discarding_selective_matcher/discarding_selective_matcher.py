from pathlib import Path
from typing import List, Tuple
from erllm import RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import find_matching_csv

from erllm.llm_matcher.evalrun import read_run_raw


def discarder(
    fraction: float, runFile: Path, similaritiesFile: Path, sim_function="overlap"
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pass


def discarding_selective_matcher(
    fraction: float, runFile: Path, similaritiesFile: Path, sim_function="overlap"
):
    completions = read_run_raw(runFile)
    dataset_ids: Tuple[Tuple[int, int], ...] = tuple(completions.keys())
    print(dataset_ids)


def get_similarities(similaritiesFile: Path, dataset_ids: Tuple[Tuple[int, int], ...]):
    pass


if __name__ == "__main__":
    # example run for debugging
    CONFIGURATIONS = {
        "base": {
            "runfiles": RUNS_FOLDER_PATH / "35_base",
            "similarities": SIMILARITIES_FOLDER_PATH,
        },
    }
    for path in CONFIGURATIONS["base"]["runfiles"].glob(
        "*dblp_scholar*force-gpt*.json"
    ):
        dataset_name = path.stem.split("-")[0]
        simPath = find_matching_csv(
            path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
        )
        if not simPath:
            raise ValueError(
                f"No matching similarity file in {CONFIGURATIONS['base']['similarities']} found for {path}"
            )
        print(discarding_selective_matcher(0.3, path, simPath))
