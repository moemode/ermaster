from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from erllm.llm_matcher.evalrun import read_run_alternate
import numpy as np


def label_k_most_uncertain(truths, predictions, probabilities, k):
    # Get the indices that would sort the probabilities array
    sorted_indices = np.argsort(probabilities)

    # Permute truths, predictions, and probabilities
    truths = truths[sorted_indices]
    predictions = predictions[sorted_indices]
    probabilities = probabilities[sorted_indices]

    # Set the first k entries of predictions to truths
    predictions[:k] = truths[:k]

    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return prec, rec, f1, acc


def label_k_most_uncertain_negative(truths, predictions, probabilities, k):
    # Get the indices of false predictions with the smallest probabilities
    false_indices = np.where(predictions == False)[0]
    sorted_false_indices = false_indices[np.argsort(probabilities[false_indices])]

    # Set the first k entries of predictions to the negative of truths
    predictions[sorted_false_indices[:k]] = truths[sorted_false_indices[:k]]

    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return prec, rec, f1, acc


def label_k_random(truths, predictions, k):
    random_indices = np.random.choice(len(predictions), k, replace=False)
    predictions[random_indices] = truths[random_indices]
    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return prec, rec, f1, acc


CONFIGURATIONS = {
    "base": {"runfiles": "runs/35_base/*force-gpt*.json", "similarities": "eval"},
}

if __name__ == "__main__":
    results = []
    for path in Path(".").glob(CONFIGURATIONS["base"]["runfiles"]):
        dataset_name = path.stem.split("-")[0]

        truths, predictions, _, probabilities, _ = read_run_alternate(path)
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

        # Create a dictionary with the results
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
    # df_results.to_csv("comparison_results.csv", index=False)
