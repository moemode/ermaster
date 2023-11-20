import json
from pathlib import Path
from typing import Iterable
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from utils import (
    NumpyEncoder,
    bernoulli_entropy,
    negative_predictive_value,
)

import pandas as pd


def read_run(run: Path):
    Path("eval").mkdir(parents=True, exist_ok=True)
    truths = []
    predictions = []
    entropies = []
    with open(run, "r") as file:
        data = json.load(file)
    for sample in data:
        prompt, truth, completion = sample["p"], sample["t"], sample["c"]
        topprobs_first = completion["logprobs"]["top_logprobs"][0]
        lp_yes = topprobs_first["Yes"] if "Yes" in topprobs_first else -1000
        lp_no = topprobs_first["No"] if "No" in topprobs_first else -1000
        p_yes, p_no = np.exp(lp_yes), np.exp(lp_no)
        entropies.append(bernoulli_entropy(p_yes / (p_yes + p_no)))
        truths.append(truth)
        predictions.append(lp_yes > lp_no)
    return truths, predictions, entropies


def eval(run: Path):
    truths, predictions, entropies = (np.array(l) for l in read_run(run))
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

    with open(Path("eval") / run.name, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(results)
    return results


def eval_dir(json_files: Iterable[Path], fname="results.csv"):
    all_results = []  # List to store results for each file
    for file in json_files:
        ds, prompt_type, model, description = file.parts[-1].split("-")
        results = eval(file)
        results.update(
            {
                "Dataset": ds,
                "PromptType": prompt_type,
                "Model": model,
                "Description": description,
            }
        )
        all_results.append(results)  # Append the results dictionary for each file

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(f"eval_writeup/{fname}", index=False)
    print(df)


if __name__ == "__main__":
    CONFIGURATIONS = {
        "base": "*force-gpt*.json",
        "hash": "*force_hash-gpt*.json",
        "base_hash": "*.json",
    }
    eval_dir(
        Path("/home/v/coding/ermaster/runs").glob(CONFIGURATIONS["base"]),
        fname="base.csv",
    )
    """
    eval(
        Path(
            "/home/v/coding/ermaster/runs/dbpedia10k-2_1250-general_complex_force_hash-gpt_3.5_turbo_instruct-1max_token_0.json"
            # "/home/v/coding/ermaster/runs/dbpedia10k-2_1250_general_complex_force-gpt-3.5_turbo_instruct-1max_token_0.json"
        )
    )
    """
