"""
Methods for reading run files, deriving classification decisions, and calculating classification and calibration metrics 
"""
import json
from pathlib import Path
from typing import Any, Dict
from attr import dataclass
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from erllm.utils import (
    NumpyEncoder,
    bernoulli_entropy,
    negative_predictive_value,
)
from erllm.calibration.reliability_diagrams import *


@dataclass
class CompletedPrompt:
    id0: int
    id1: int
    prompt_string: str
    truth: bool
    completion: Dict[str, Any]
    duration: float
    entropy: float = None
    prediction: Any = None
    probability: float = None

    @classmethod
    def from_json(cls, sample: Dict[str, Any]) -> "CompletedPrompt":
        prompt_string, truth, completion = sample["p"], sample["t"], sample["c"]
        id0, id1 = sample["id0"], sample["id1"]
        topprobs_first = completion["logprobs"]["top_logprobs"][0]
        # Define Yes/No tokens
        yn_tokens = ["Yes", "No", " Yes", " No"]
        # Initialize dictionary for Yes/No probabilities
        yn_probs = {}
        total_probability = 0
        # Sum the probabilities of Yes/No tokens
        for t in yn_tokens:
            yn_probs[t] = np.exp(topprobs_first.get(t, -1000))
            total_probability += yn_probs[t]
        p_yes = (yn_probs["Yes"] + yn_probs[" Yes"]) / total_probability
        p_no = (yn_probs["No"] + yn_probs[" No"]) / total_probability
        # Find the token with the maximum probability
        # max_prob_token = max(yn_probs, key=yn_probs.get)
        # Calculate the ratio of the max_prob_token probability to the total probability
        # probability = yn_probs[max_prob_token] / total_probability
        entropy = bernoulli_entropy(p_yes / (p_yes + p_no))
        prediction = p_yes > p_no
        probability = p_yes if p_yes > p_no else p_no
        return cls(
            id0=sample["id0"],
            id1=sample["id1"],
            prompt_string=prompt_string,
            truth=truth,
            completion=completion,
            duration=sample["d"],  #
            entropy=entropy,
            prediction=prediction,
            probability=probability,
        )


def calibration_data(truths, predictions, probabilities):
    """
    Calculate calibration metrics given ground truth, predictions, and predicted probabilities.
    These are returned by read_run.

    Parameters:
        truths (array-like): Ground truth labels.
        predictions (array-like): Predicted binary labels.
        probabilities (array-like): Predicted probabilities.

    Returns:
        dict: Calibration metrics including Brier Score, Expected Calibration Error (ECE), and confusion matrix components.
    """
    probabilities_brier = probabilities.copy()
    pred0 = 0 == predictions
    probabilities_brier[pred0] = 1 - probabilities_brier[pred0]
    brier = brier_score_loss(truths, probabilities_brier)
    ece = compute_calibration(truths, predictions, probabilities, num_bins=10)[
        "expected_calibration_error"
    ]
    probs_for_truth = np.where(predictions == truths, probabilities, 1 - probabilities)
    average_calibration_error = np.mean(1 - probs_for_truth)
    # create an estimated confusion matrix from predictions and probabilities
    est_tn = np.sum((~predictions) * probabilities)
    est_fp = np.sum((~predictions) * (1 - probabilities))
    est_fn = np.sum(predictions * (1 - probabilities))
    est_tp = np.sum(predictions * probabilities)
    confusion_matrix = np.array([[est_tn, est_fp], [est_fn, est_tp]])
    accuracy = (est_tp + est_tn) / np.sum(confusion_matrix)
    precision = est_tp / (est_tp + est_fp) if (est_tp + est_fp) > 0 else 0
    recall = est_tp / (est_tp + est_fn) if (est_tp + est_fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "Brier Score": brier,
        "ECE": ece,
        "ACE": average_calibration_error,
        "EST_TN": est_tn,
        "EST_FP": est_fp,
        "EST_FN": est_fn,
        "EST_TP": est_tp,
        "EST_Accuracy": accuracy,
        "EST_Precision": precision,
        "EST_Recall": recall,
        "EST_F1": f1,
    }


def read_run_raw(run: Path) -> Dict[tuple, CompletedPrompt]:
    """
    Read a JSON run file and create a dictionary mapping (id0, id1) tuples to CompletedPrompt objects.

    Parameters:
        run (Path): Path to the JSON run file.

    Returns:
        Dict: Mapping of (id0, id1) tuples to CompletedPrompt objects.
    """
    with open(run, "r") as file:
        data = json.load(file)
    # Create a dictionary mapping a tuple of (id0, id1) to CompletedPrompt objects
    prompt_dict: Dict[tuple, CompletedPrompt] = {
        (item["id0"], item["id1"]): CompletedPrompt.from_json(item) for item in data
    }
    return prompt_dict


def read_run(
    run: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Read a JSON run file, process the completed prompts, and return relevant information for evaluation.

    Parameters:
        run (Path): Path to the JSON run file.

    Returns:
        Tuple: A tuple containing arrays of truths, predictions, entropies, probabilities, and pairs.
    """
    pairs = []
    truths = []
    predictions = []
    entropies = []
    probabilities = []
    with open(run, "r") as file:
        data = json.load(file)
    for sample in data:
        _, truth, completion = sample["p"], sample["t"], sample["c"]
        pairs.append((sample["id0"], sample["id1"]))
        topprobs_first = completion["logprobs"]["top_logprobs"][0]
        # Define Yes/No tokens
        yn_tokens = ["Yes", "No", " Yes", " No"]
        # Initialize dictionary for Yes/No probabilities
        yn_probs = {}
        total_probability = 0
        # Sum the probabilities of Yes/No tokens
        for t in yn_tokens:
            yn_probs[t] = np.exp(topprobs_first.get(t, -1000))
            total_probability += yn_probs[t]
        p_yes = (yn_probs["Yes"] + yn_probs[" Yes"]) / total_probability
        p_no = (yn_probs["No"] + yn_probs[" No"]) / total_probability
        # Find the token with the maximum probability
        # max_prob_token = max(yn_probs, key=yn_probs.get)
        # Calculate the ratio of the max_prob_token probability to the total probability
        # probability = yn_probs[max_prob_token] / total_probability
        entropies.append(bernoulli_entropy(p_yes / (p_yes + p_no)))
        truths.append(truth)
        predictions.append(p_yes > p_no)
        probabilities.append(p_yes if p_yes > p_no else p_no)
    return (
        np.array(truths),
        np.array(predictions),
        np.array(entropies),
        np.array(probabilities),
        pairs,
    )


def eval(run: Path, save_to: Path):
    """
    Evaluate LLM matcher performance based on a run file and save the results.

    Parameters:
        run (Path): Path to the JSON run file produced by gpt.py.

    Returns:
        Dict: Evaluation results.
    """
    truths, predictions, entropies, probabilities, _ = read_run(run)
    truths = truths.astype(bool)
    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
    tp_ind = np.logical_and(truths, predictions)
    tn_ind = np.logical_and(~truths, ~predictions)
    fp_ind = np.logical_and(~truths, predictions)
    fn_ind = np.logical_and(truths, ~predictions)

    tp = np.sum(tp_ind)
    tn = np.sum(tn_ind)
    fp = np.sum(fp_ind)
    fn = np.sum(fn_ind)

    assert (tn, fp, fn, tp) == tuple(confusion_matrix(truths, predictions).ravel())

    correct = truths == predictions
    correct_entropies, wrong_entropies = entropies[correct], entropies[~correct]
    tn_entropies = entropies[tn_ind]
    fp_entropies = entropies[fp_ind]
    fn_entropies = entropies[fn_ind]
    tp_entropies = entropies[tp_ind]
    results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "NPV": negative_predictive_value(tn, fn),
        "Correct Entropy": correct_entropies.mean(),
        "Wrong Entropy": wrong_entropies.mean(),
        "TP Entropy": tp_entropies.mean(),
        "TN Entropy": tn_entropies.mean(),
        "FP Entropy": fp_entropies.mean(),
        "FN Entropy": fn_entropies.mean(),
    }
    calibration_results = calibration_data(truths, predictions, probabilities)
    results.update(calibration_results)
    with open(save_to / run.name, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(results)
    return results
