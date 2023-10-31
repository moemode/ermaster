import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import bernoulli_entropy


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
    precision = precision_score(truths, predictions)
    recall = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    accuracy = accuracy_score(truths, predictions)
    correct = truths == predictions
    correct_entropies, wrong_entropies = entropies[correct], entropies[~correct]
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Correct Entropy": correct_entropies.mean(),
        "Wrong Entropy": wrong_entropies.mean(),
    }
    with open(Path("eval") / run.name, "w") as f:
        json.dump(results, f, indent=2)
    print(results)


if __name__ == "__main__":
    eval(
        Path(
            "/home/v/coding/ermaster/runs/beer_simple_gpt-3.5-turbo-instruct_1max_token_0.json"
        )
    )
