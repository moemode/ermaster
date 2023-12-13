from pathlib import Path
from typing import Iterable, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from cost import str_cost
from erllm.eval.evalrun import read_run_alternate, read_run_raw
import pandas as pd


def pre_llm(
    threshold: float, runFile: Path, similaritiesFile: Path, sim_function="overlap"
):
    truths, predictions, entropies, probabilities, pair_ids = read_run_alternate(
        runFile
    )
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
    json_dataset_name = run_file.stem.split("-")[0]
    json_dataset_name = json_dataset_name.replace("_1250", "")
    matching_csv = next(
        (
            csv_file
            for csv_file in similarity_files
            if json_dataset_name in csv_file.stem
        ),
        None,
    )
    return matching_csv


if __name__ == "__main__":
    CONFIGURATIONS = {
        "base": {
            "runfiles": "runs/35_base/*dblp_scholar*force-gpt*.json",
            "similarities": "eval",
        },
    }
    for path in Path(".").glob(CONFIGURATIONS["base"]["runfiles"]):
        dataset_name = path.stem.split("-")[0]
        simPath = find_matching_csv(
            path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
        )
        if not simPath:
            raise ValueError(
                f"No matching similarity file in {CONFIGURATIONS['base']['similarities']} found for {path}"
            )
        pre_llm(0.3, path, simPath)
