"""
Defines functions to manually label predictions by selecting the k most uncertain, 
k most uncertain negative, and k random predictions from a given set.
It applies these labeling strategies to predictions on different datasets, 
calculates various classification metrics, and saves the results for comparison.
"""
from typing import Tuple
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import read_run
import numpy as np
from erllm.utils import classification_metrics


def label_k_most_uncertain(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, k: int
) -> Tuple[float, float, float, float]:
    """
    Label the k most uncertain predictions and calculate classification metrics.

    Parameters:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Model predictions.
        probabilities (np.ndarray): Prediction probabilities.
        k (int): Number of most uncertain predictions to label.

    Returns:
        Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
    """
    # Get the indices sorted ascendingly by associated probabilities
    sorted_indices = np.argsort(probabilities)
    # Permute truths, predictions, and probabilities
    truths = truths[sorted_indices]
    predictions = predictions[sorted_indices]
    probabilities = probabilities[sorted_indices]
    # Set the first k entries of predictions to truths
    predictions[:k] = truths[:k]
    return classification_metrics(truths, predictions)


def label_k_most_uncertain_negative(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, k: int
) -> Tuple[float, float, float, float]:
    """
    Label the k most uncertain negative predictions and calculate classification metrics.

    Parameters:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Model predictions.
        probabilities (np.ndarray): Prediction probabilities.
        k (int): Number of most uncertain negative predictions to label.

    Returns:
        Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
    """
    # Get the indices of false predictions with the smallest probabilities
    false_indices = np.where(predictions == False)[0]
    sorted_false_indices = false_indices[np.argsort(probabilities[false_indices])]
    # Set the first k entries of predictions to the negative of truths
    predictions[sorted_false_indices[:k]] = truths[sorted_false_indices[:k]]
    return classification_metrics(truths, predictions)


def label_k_random(
    truths: np.ndarray, predictions: np.ndarray, k: int
) -> Tuple[float, float, float, float]:
    """
    Label k random predictions and calculate classification metrics.

    Parameters:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Model predictions.
        k (int): Number of random predictions to label.

    Returns:
        Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
    """
    random_indices = np.random.choice(len(predictions), k, replace=False)
    predictions[random_indices] = truths[random_indices]
    return classification_metrics(truths, predictions)


CONFIGURATIONS = {
    "base": {"runfolder": RUNS_FOLDER_PATH / "35_base", "outfile_name": "base.csv"},
}

MLABELING_FOLDER = EVAL_FOLDER_PATH / "manual_labeling"

if __name__ == "__main__":
    MLABELING_FOLDER.mkdir(parents=True, exist_ok=True)
    results = []
    cfg = CONFIGURATIONS["base"]
    # iterate over datasets
    for path in cfg["runfolder"].glob("*force-gpt*.json"):
        dataset_name = path.stem.split("-")[0]
        truths, predictions, _, probabilities, _ = read_run(path)
        f1 = f1_score(truths, predictions)
        rec = recall_score(truths, predictions)
        k = int(round(0.05 * len(truths)))
        # Labeling most uncertain
        (
            prec_uncertain,
            rec_uncertain,
            f1_uncertain,
            acc_uncertain,
        ) = label_k_most_uncertain(truths, predictions, probabilities, k)

        # Labeling randomly
        prec_random, rec_random, f1_random, acc_random = label_k_random(
            truths, predictions, k
        )

        # Create a dictionary with the results for the individual dataset
        result_dict = {
            "Dataset": dataset_name,
            "F1": f1,
            "Recall": rec,
            "Precision_Uncertain": prec_uncertain,
            "Recall_Uncertain": rec_uncertain,
            "F1_Uncertain": f1_uncertain,
            "Accuracy_Uncertain": acc_uncertain,
            "Precision_Random": prec_random,
            "Recall_Random": rec_random,
            "F1_Random": f1_random,
            "Accuracy_Random": acc_random,
        }

        # Append the dictionary to the results list
        results.append(result_dict)

    # Create a dataframe from the list of dictionaries
    df_results = pd.DataFrame(results)

    # Display or save the dataframe as needed
    print(
        df_results[
            [
                "Dataset",
                "F1",
                "F1_Uncertain",
                "F1_Random",
            ]
        ]
    )
    df_results.to_csv(MLABELING_FOLDER / cfg["outfile_name"], index=False)
