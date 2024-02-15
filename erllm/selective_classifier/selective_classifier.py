"""
Run and evaluate selective classification.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List
from erllm import RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import CompletedPrompt, read_run_raw
from erllm.utils import classification_metrics


def selective_classifier(
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
        # Filter rows based on the specified threshold for the confidence
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


def selective_classifier_cov(
    runFile: Path, coverages: Iterable[float]
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
    completed_prompts = completions.values()
    # sort completed_prompts descendingly in order of confidence
    completed_prompts = sorted(
        completed_prompts, key=lambda cp: cp.probability, reverse=True
    )
    data = []
    for cov in coverages:
        # Filter rows based on the specified threshold for the "overlap" column
        covered = completed_prompts[: int(cov * len(completed_prompts))]
        threshold = covered[-1].probability if covered else 1.0
        if covered:
            truth_preds = [(cp.truth, cp.prediction) for cp in covered]
            truths, predictions = zip(*truth_preds)
        else:
            truths, predictions = [], []
        prec, rec, f1, acc = classification_metrics(truths, predictions)
        data.append(
            {
                "threshold": threshold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "coverage": cov,
            }
        )
    return data


if __name__ == "__main__":
    CONFIGURATIONS = {
        "base": {
            "runfiles": RUNS_FOLDER_PATH / "35_base",
        },
    }
    for path in CONFIGURATIONS["base"]["runfiles"].glob(
        "*dblp_scholar*force-gpt*.json"
    ):
        dataset_name = path.stem.split("-")[0]
        print(selective_classifier(path, [0.3, 0.5, 0.7]))
