"""
Defines functions to manually label predictions by selecting the k most uncertain, 
k most uncertain negative, and k random predictions from a given set.
It applies these labeling strategies to predictions on different datasets, 
calculates various classification metrics, and saves the results for comparison.
"""
from typing import Dict, Tuple
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH
import numpy as np
from erllm.llm_matcher.evalrun import read_run
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
    truths = truths.copy()
    predictions = predictions.copy()
    probabilities = probabilities.copy()
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
    truths = truths.copy()
    predictions = predictions.copy()
    probabilities = probabilities.copy()
    # Get the indices of false predictions with the smallest probabilities
    false_indices = np.where(predictions == False)[0]
    sorted_false_indices = false_indices[np.argsort(probabilities[false_indices])]
    # Set the first k entries of predictions to the negative of truths
    predictions[sorted_false_indices[:k]] = truths[sorted_false_indices[:k]]
    return classification_metrics(truths, predictions)


def label_k_random(
    truths: np.ndarray, predictions: np.ndarray, k: int, tries: int
) -> Dict[str, np.floating]:
    """
    Label k random predictions and calculate classification metrics.

    Parameters:
        truths (np.ndarray): Ground truth labels.
            The array containing the ground truth labels for the predictions.
        predictions (np.ndarray): Model predictions.
            The array containing the predicted labels.
        k (int): Number of random predictions to label.
            The number of random predictions to label and evaluate.
        tries (int, optional): Number of tries to perform the labeling and evaluation process.

    Returns:
        dict: A dictionary containing the calculated classification metrics.
            The dictionary contains the following keys:
            - "Precision_Random": Mean precision of the random predictions.
            - "Recall_Random": Mean recall of the random predictions.
            - "F1_Random": Mean F1 score of the random predictions.
            - "Accuracy_Random": Mean accuracy of the random predictions.
            - "Precision_Random_std": Standard deviation of precision for the random predictions.
            - "Recall_Random_std": Standard deviation of recall for the random predictions.
            - "F1_Random_std": Standard deviation of F1 score for the random predictions.
            - "Accuracy_Random_std": Standard deviation of accuracy for the random predictions.
    """
    truths = truths.copy()
    predictions = predictions.copy()
    precisions, recalls, f1s, accuracies = [], [], [], []
    for i in range(tries):
        random_indices = np.random.choice(len(predictions), k, replace=False)
        predictions[random_indices] = truths[random_indices]
        prec, rec, f1, acc = classification_metrics(truths, predictions)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        accuracies.append(acc)
    return {
        "Precision_Random": np.mean(precisions),
        "Recall_Random": np.mean(recalls),
        "F1_Random": np.mean(f1s),
        "Accuracy_Random": np.mean(accuracies),
        "Precision_Random_std": np.std(precisions),
        "Recall_Random_std": np.std(recalls),
        "F1_Random_std": np.std(f1s),
        "Accuracy_Random_std": np.std(accuracies),
    }


CONFIGURATIONS = {
    "base": {
        "runfolder": RUNS_FOLDER_PATH / "35_base",
        "outfile_name": "base.csv",
        "tries": 30,
        "fractions": [0, 0.05, 0.1, 0.15, 0.2],
    },
    "gpt-4-base": {
        "runfolder": RUNS_FOLDER_PATH / "4_base",
        "outfile_name": "4-base.csv",
        "tries": 30,
        "fractions": [0, 0.05, 0.1, 0.15, 0.2],
    },
}

MLABELING_FOLDER = EVAL_FOLDER_PATH / "manual_labeling"

if __name__ == "__main__":
    MLABELING_FOLDER.mkdir(parents=True, exist_ok=True)
    results = []
    cfg = CONFIGURATIONS["gpt-4-base"]
    # iterate over datasets
    for path in cfg["runfolder"].glob("*force-gpt*.json"):
        dataset_name = path.stem.split("-")[0]
        truths, predictions, _, probabilities, _ = read_run(path)
        f1 = f1_score(truths, predictions)
        rec = recall_score(truths, predictions)
        orig_predictions = predictions.copy()
        for f in cfg["fractions"]:
            k = int(round(f * len(truths)))
            # Labeling most uncertain
            (
                prec_uncertain,
                rec_uncertain,
                f1_uncertain,
                acc_uncertain,
            ) = label_k_most_uncertain(truths, predictions, probabilities, k)
            # Create a dictionary with the results for the individual dataset
            result_dict = {
                "Dataset": dataset_name,
                "F1": f1,
                "Fraction": f,
                "Recall": rec,
                "Precision_Uncertain": prec_uncertain,
                "Recall_Uncertain": rec_uncertain,
                "F1_Uncertain": f1_uncertain,
                "Accuracy_Uncertain": acc_uncertain,
                **label_k_random(truths, predictions, k, cfg["tries"]),
            }
            assert all(predictions == orig_predictions)
            # Append the dictionary to the results list
            results.append(result_dict)
    # Create a dataframe from the list of dictionaries
    df_results = pd.DataFrame(results)
    print(
        df_results[
            [
                "Dataset",
                "Fraction",
                "F1",
                "F1_Uncertain",
                "F1_Random",
                "F1_Random_std",
            ]
        ]
    )
    df_results.to_csv(MLABELING_FOLDER / cfg["outfile_name"], index=False)
