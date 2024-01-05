"""
Defines functions to manually label predictions by selecting the k most uncertain, 
k most uncertain negative, and k random predictions from a given set.
It applies these labeling strategies to predictions on different datasets, 
calculates various classification metrics, and saves the results for comparison.
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


def eval_label_k_random(
    truths: np.ndarray, predictions: np.ndarray, k: int, tries: int
) -> dict:
    metrics = []
    for _ in range(tries):
        corrected_predictions = label_k_random(truths, predictions, k)
        metrics.append(
            eval_corrected_prediction(truths, predictions, corrected_predictions)
        )
    return dict(zip(eval_corrected_prediction.ret_dict_keys, np.mean(metrics, axis=0)))


def eval_selective_matcher(
    truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, k: int
) -> dict:
    corrected_predictions = selective_matcher(truths, predictions, probabilities, k)
    return dict(
        zip(
            eval_corrected_prediction.ret_dict_keys,
            eval_corrected_prediction(truths, predictions, corrected_predictions),
        )
    )


def blah():
    """
    truths = truths_orig.copy()
    predictions = predictions_orig.copy()
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
        "N_corrected_Random": np.sum(predictions != predictions_orig),
        "F1_Random_std": np.std(f1s),
        "Accuracy_Random_std": np.std(accuracies),
    }
    """
    pass


# def label_k_most_uncertain_negative(
#     truths: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, k: int
# ) -> Tuple[float, float, float, float]:
#     """
#     Label the k most uncertain negative predictions and calculate classification metrics.

#     Parameters:
#         truths (np.ndarray): Ground truth labels.
#         predictions (np.ndarray): Model predictions.
#         probabilities (np.ndarray): Prediction probabilities.
#         k (int): Number of most uncertain negative predictions to label.

#     Returns:
#         Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
#     """
#     truths = truths.copy()
#     predictions = predictions.copy()
#     probabilities = probabilities.copy()
#     # Get the indices of false predictions with the smallest probabilities
#     false_indices = np.where(predictions == False)[0]
#     sorted_false_indices = false_indices[np.argsort(probabilities[false_indices])]
#     # Set the first k entries of predictions to the negative of truths
#     predictions[sorted_false_indices[:k]] = truths[sorted_false_indices[:k]]
#     return classification_metrics(truths, predictions)
