from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from erllm import RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.llm_matcher.evalrun import CompletedPrompt, read_run_raw
from erllm.utils import classification_metrics


def selective_matcher(
    runFile: Path, thresholds: Iterable[float]
) -> List[Dict[str, Any]]:
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
    completions: Dict[tuple, CompletedPrompt] = read_run_raw(runFile)
    data = []
    for threshold in thresholds:
        # Filter rows based on the specified threshold for the "overlap" column
        above_th: Dict[tuple, CompletedPrompt] = {
            pair_id: cp
            for pair_id, cp in completions.items()
            if cp.probability > threshold
        }
        if above_th:
            truth_preds = [(cp.truth, cp.prediction) for cp in above_th.values()]
            truths, predictions = zip(*truth_preds)
        else:
            truths, predictions = [], []
        prec, rec, f1, acc = classification_metrics(truths, predictions)
        N = len(completions)
        N_rem = len(above_th)
        coverage = N_rem / N
        data.append(
            {
                "threshold": threshold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "coverage": coverage,
            }
        )
    return data


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
        print(selective_matcher(path, [0.3, 0.5, 0.7]))
