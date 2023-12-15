import json
import math
import random
import time
from pathlib import Path

import numpy as np
import openai
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tiktoken
from tqdm import tqdm


def write_json_iter(it, fh, N=None):
    print("[", file=fh)
    for n, rec in tqdm(enumerate(it), total=N):
        if n > 0:
            print(",", file=fh)
        json.dump(rec, fh)
        fh.flush()
    print("\n]", file=fh)


def numbered_path(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # Find the lowest untaken number for the filename
    base_path, file_stem = p.parent, p.stem
    number = 0
    while (base_path / Path(f"{file_stem}_{number}{p.suffix}")).exists():
        number += 1
    return base_path / Path(f"{file_stem}_{number}{p.suffix}")


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def bernoulli_entropy(p):
    # Calculate entropy
    if p == 0 or p == 1:
        return 0  # Entropy is 0 if p is 0 or 1
    else:
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def recall(tp, fn):
    return tp / (tp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def negative_predictive_value(tn, fn):
    return tn / (tn + fn)


def f1(tp, tn, fp, fn):
    p = precision(tn, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def num_tokens_from_string(string: str, model: str):
    """Returns the number of tokens in a text string."""
    if not hasattr(num_tokens_from_string, "cached_encodings"):
        num_tokens_from_string.cached_encodings = {}
    if model not in num_tokens_from_string.cached_encodings:
        num_tokens_from_string.cached_encodings[model] = tiktoken.encoding_for_model(
            model
        )
    encoding = num_tokens_from_string.cached_encodings[model]
    num_tokens = len(encoding.encode(string))
    return num_tokens


def rename_datasets(df, preserve_sampled=True):
    # Remove prefixes from Dataset column
    df["Dataset"] = df["Dataset"].str.replace(r"^structured_|textual_", "", regex=True)
    # Reorder columns for clarity
    df["Dataset"] = df["Dataset"].str.replace("_", "-").str.title()
    df["Dataset"] = df["Dataset"].str.replace(
        "-1250", " Sampled" if preserve_sampled else ""
    )
    df["Dataset"] = df["Dataset"].str.replace("Dbpedia10K", "DBpedia")
    return df


def classification_metrics(
    truths: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float, float]:
    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return prec, rec, f1, acc
