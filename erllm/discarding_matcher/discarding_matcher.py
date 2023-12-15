from pathlib import Path
from typing import Iterable, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import pandas as pd
from erllm import RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.llm_matcher.cost import str_cost
from erllm.llm_matcher.evalrun import read_run, read_run_raw


def discarding_matcher(
    threshold: float, runFile: Path, similaritiesFile: Path, sim_function="overlap"
):
    """
    Evaluate the performance of a discarding matcher based on a given threshold.

    Parameters:
        threshold (float): The similarity threshold for discarding pairs.
        runFile (Path): Path to the run JSON file containing completion information.
        Created by llm_matcher/gpt.py.
        similaritiesFile (Path): Path to the CSV file containing pair similarities.
        Created by discarder/discarder.py.
        sim_function (str, optional): The similarity function to use. Default is "overlap".

    Returns:
        tuple: A tuple containing accuracy, precision, recall, F1 score,
               remaining cost, cost reduction percentage, remaining duration,
               and duration reduction percentage.
    """
    truths, predictions, _, _, pair_ids = read_run(runFile)
    completions = read_run_raw(runFile)
    prompts = list(map(lambda cp: cp.prompt_string, completions.values()))
    cost_full = str_cost(prompts, 1, "gpt-3.5-turbo-instruct")
    duration_full = sum(map(lambda cp: cp.duration, completions.values()))
    similarities = pd.read_csv(similaritiesFile)
    # ensure all pairs have a similarity value
    sim_pairs = set(zip(similarities["table1.id"], similarities["table2.id"]))
    if not set(pair_ids).issubset(sim_pairs):
        raise ValueError("Similarity file misses values for some pairs.")
    # Filter rows based on the specified threshold for the "overlap" column
    discarded_rows = similarities[similarities[sim_function] <= threshold]
    discarded_pairs = set(zip(discarded_rows["table1.id"], discarded_rows["table2.id"]))
    if threshold == 0.0:
        discarded_pairs = set()
    # discarded prompts
    discarded = [
        completions[pair_id] for pair_id in discarded_pairs if pair_id in pair_ids
    ]
    discarded_prompts = list(map(lambda cp: cp.prompt_string, discarded))
    cost_remaining = cost_full - str_cost(
        discarded_prompts, 1, "gpt-3.5-turbo-instruct"
    )
    duration_remaining = duration_full - sum(map(lambda cp: cp.duration, discarded))
    # Get indices of discarded pairs in pairs
    discarded_pairs_indices = [
        i for i, pair in enumerate(pair_ids) if pair in discarded_pairs
    ]
    predictions[discarded_pairs_indices] = 0
    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return (
        acc,
        prec,
        rec,
        f1,
        cost_remaining,
        cost_remaining / cost_full,
        duration_remaining,
        duration_remaining / duration_full,
    )


def find_matching_csv(
    run_file: Path, similarity_files: Iterable[Path]
) -> Optional[Path]:
    """
    Find the matching similarity CSV file for a given run JSON file based on dataset name.
    Works only if the dataset name is the first part of the run JSON file name
    and contained in the similarity CSV file name.

    Parameters:
        run_file (Path): The path to the run JSON file.
        similarity_files (Iterable[Path]): Iterable of paths to similarity CSV files.

    Returns:
        Optional[Path]: The path to the matching similarity CSV file, or None if not found.
    """
    ds_name = run_file.stem.split("-")[0]
    ds_name = ds_name.replace("_1250", "")
    matching_csv = next(
        (csv_file for csv_file in similarity_files if ds_name in csv_file.stem),
        None,
    )
    return matching_csv


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
        print(discarding_matcher(0.3, path, simPath))
