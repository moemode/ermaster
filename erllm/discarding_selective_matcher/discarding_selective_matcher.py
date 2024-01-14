from pathlib import Path
from typing import List, Set, Tuple
import pandas as pd
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
    dataset_ids: Set[Tuple[int, int]] = set(completions.keys())
    sims = get_similarities(dataset_ids, fraction, similaritiesFile)
    discarder(sims, fraction, sim_function)


def get_similarities(
    dataset_ids: set[Tuple[int, int]],
    similarities_file: Path,
):
    similarities = pd.read_csv(similarities_file)
    sim_pairs = set(zip(similarities["table1.id"], similarities["table2.id"]))
    if not set(dataset_ids).issubset(sim_pairs):
        raise ValueError("Similarity file misses values for some pairs.")
    # Filter rows to only include those with IDs present in dataset_ids
    similarities = similarities[
        similarities.apply(
            lambda row: (row["table1.id"], row["table2.id"]) in dataset_ids, axis=1
        )
    ]
    if not len(similarities) == len(dataset_ids):
        raise ValueError(
            f"Similarity file has {len(similarities)} rows, but {len(dataset_ids)} pairs were expected."
        )
    return similarities


def discarder(similarities: pd.DataFrame, fraction: float, sim_function="overlap"):
    if not sim_function in similarities.columns:
        raise ValueError(
            f"Similarity file does not have a column named {sim_function}."
        )
    similarities_sorted = similarities.sort_values(by=[sim_function])
    n_discard = int(len(similarities) * fraction)
    discarded_rows = similarities_sorted.head(n_discard)
    kept_rows = similarities_sorted.tail(len(similarities) - n_discard)
    return discarded_rows, kept_rows


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
