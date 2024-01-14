from pathlib import Path
from typing import Set, Tuple
import pandas as pd
from erllm import RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import find_matching_csv
from erllm.llm_matcher.evalrun import CompletedPrompt, read_run_raw


def discarding_selective_matcher(
    discard_fraction: float,
    label_fraction: float,
    runFile: Path,
    similaritiesFile: Path,
    sim_function="overlap",
):
    completions = read_run_raw(runFile)
    dataset_pair_ids: Set[Tuple[int, int]] = set(completions.keys())
    sims = get_similarities(dataset_pair_ids, similaritiesFile)
    n_discard = int(len(dataset_pair_ids) * discard_fraction)
    discarded, kept = discarder(sims, n_discard, sim_function)
    discarded_pairs = set(zip(discarded["table1.id"], discarded["table2.id"]))
    kept_pairs = set(zip(kept["table1.id"], kept["table2.id"]))
    kept_completions = {
        pair_id: cp for pair_id, cp in completions.items() if pair_id in kept_pairs
    }
    predictions, llm_pairs, human_label_pairs = selective_matcher(
        label_fraction, kept_completions
    )
    predictions.update({pair_id: False for pair_id in discarded_pairs})
    assert len(predictions) == len(dataset_pair_ids)
    assert len(discarded_pairs) + len(llm_pairs) + len(human_label_pairs) == len(
        dataset_pair_ids
    )
    return predictions, discarded_pairs, llm_pairs, human_label_pairs


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


def discarder(similarities: pd.DataFrame, n_discard: int, sim_function="overlap"):
    if not sim_function in similarities.columns:
        raise ValueError(
            f"Similarity file does not have a column named {sim_function}."
        )
    similarities_sorted = similarities.sort_values(by=[sim_function])
    discarded_rows = similarities_sorted.head(n_discard)
    kept_rows = similarities_sorted.tail(len(similarities) - n_discard)
    return discarded_rows, kept_rows


def selective_matcher(
    label_fraction: float, completions: dict[Tuple[int, int], CompletedPrompt]
) -> Tuple[dict[Tuple[int, int], bool], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    predictions: dict[Tuple[int, int], bool] = {
        id_pair: cp.prediction for id_pair, cp in completions.items()
    }
    n_label = int(len(completions) * label_fraction)
    # get the n_label id pairs with smallest confidence
    sorted_completions = sorted(completions.values(), key=lambda cp: cp.probability)
    least_confident_ids = set((cp.id0, cp.id1) for cp in sorted_completions[:n_label])
    llm_ids = set(completions.keys()) - least_confident_ids
    for id_pair in least_confident_ids:
        predictions[id_pair] = completions[id_pair].truth
    return predictions, llm_ids, least_confident_ids


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
        print(discarding_selective_matcher(0.3, 0.1, path, simPath))
