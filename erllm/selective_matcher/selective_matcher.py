"""
Implements the selective matcher and the labeling of randomly chosen predictions.
It applies these to predictions on different datasets and calculates various classification metrics.
"""

from typing import Tuple
from sklearn.metrics import confusion_matrix
import numpy as np
from erllm.utils import classification_metrics


def selective_matcher(
    truths: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Selective matcher where user correctly labels the k most uncertain predictions.

    Parameters:
        truths_orig (np.ndarray): Ground truth labels.
        predictions_orig (np.ndarray): Model predictions.
        probabilities_orig (np.ndarray): Prediction probabilities.
        k (int): Number of most uncertain predictions to label.

    Returns:
        Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
    """
    corrected_predictions = predictions.copy()
    # Get the indices sorted ascendingly by associated probabilities
    least_conf_indices = np.argsort(probabilities)
    corrected_predictions[least_conf_indices[:k]] = truths[least_conf_indices[:k]]
    return corrected_predictions


def label_k_random(truths: np.ndarray, predictions: np.ndarray, k: int) -> np.ndarray:
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
    corrected_predictions = predictions.copy()
    random_indices = np.random.choice(len(predictions), k, replace=False)
    corrected_predictions[random_indices] = truths[random_indices]
    return corrected_predictions


def eval_corrected_prediction(
    truths: np.ndarray,
    predictions: np.ndarray,
    corrected_predictions: np.ndarray,
) -> Tuple[float, float, float, float, int, int, int, int, int]:
    """
    Evaluate the correctness of corrected predictions compared to ground truth.

    Args:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Original predictions.
        corrected_predictions (np.ndarray): Corrected predictions.

    Returns:
        Tuple[float, float, float, float, int, int, int, int, int]: A tuple containing the precision, recall, F1 score,
        accuracy, number of corrected predictions, true negatives, false positives, false negatives, and true positives.
    """
    n_corrected = np.sum(predictions != corrected_predictions)
    prec, rec, f1, acc = classification_metrics(truths, corrected_predictions)
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(truths, corrected_predictions).ravel()
    return prec, rec, f1, acc, n_corrected, tn, fp, fn, tp


eval_corrected_prediction.ret_dict_keys = (
    "Precision",
    "Recall",
    "F1",
    "Accuracy",
    "N_corrected",
    "TN",
    "FP",
    "FN",
    "TP",
)


def eval_selective_matcher(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, k: int
) -> dict:
    """
    Evaluate the performance of the selective matcher.

    Args:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Predicted labels.
        probabilities (np.ndarray): Prediction probabilities.
        k (int): Number of most uncertain predictions to label.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    corrected_predictions = selective_matcher(truths, predictions, probabilities, k)
    return dict(
        zip(
            eval_corrected_prediction.ret_dict_keys,
            eval_corrected_prediction(truths, predictions, corrected_predictions),
        )
    )


def eval_label_k_random(
    truths: np.ndarray, predictions: np.ndarray, k: int, tries: int
) -> dict:
    """
    Evaluate the performance of the label_k_random function by comparing the corrected predictions
    with the ground truth labels.

    Args:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Predicted labels.
        k (int): Number of labels to randomly select for correction.
        tries (int): Number of times to repeat the evaluation.

    Returns:
        dict: A dictionary containing the evaluation metrics, including the mean and standard deviation
        of each metric.
    """
    metrics = []
    for _ in range(tries):
        corrected_predictions = label_k_random(truths, predictions, k)
        metrics.append(
            eval_corrected_prediction(truths, predictions, corrected_predictions)
        )
    sd_ret_dict_keys = [k + " SD" for k in eval_corrected_prediction.ret_dict_keys]
    sds = zip(sd_ret_dict_keys, np.std(metrics, axis=0))
    means = zip(eval_corrected_prediction.ret_dict_keys, np.mean(metrics, axis=0))
    return dict(list(sds) + list(means))
