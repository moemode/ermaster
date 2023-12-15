from io import TextIOWrapper
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable
import numpy as np
import openai
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tiktoken
from tqdm import tqdm


def write_json_iter(it: Iterable, fh: TextIOWrapper, N=None):
    """
    Write an iterable to a JSON file as soon as new data is available.

    Parameters:
        it (iterable): Iterable to be written to the file.
        fh (file handle): File handle to write the JSON data.
        N (int, optional): Total number of elements in the iterable (for progress tracking).

    Returns:
        None
    """
    print("[", file=fh)
    for n, rec in tqdm(enumerate(it), total=N):
        if n > 0:
            print(",", file=fh)
        json.dump(rec, fh)
        fh.flush()
    print("\n]", file=fh)


def numbered_path(p: Path) -> Path:
    """
    Generate a numbered file path by modifying p to avoid overwriting existing files.

    Parameters:
        p (Path): Original file path.

    Returns:
        Path: Numbered file path.
    """
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
) -> Callable:
    """
    3rd party code to retry a function with exponential backoff.
    Made available by OpenAI at https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    """

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


def bernoulli_entropy(p: float) -> float:
    """
    Calculate the entropy of a Bernoulli distribution.

    Parameters:
        p (float): Probability of success.

    Returns:
        float: Entropy value.
    """
    # Calculate entropy
    if p == 0 or p == 1:
        return 0  # Entropy is 0 if p is 0 or 1
    else:
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate accuracy.

    Parameters:
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: Accuracy value.
    """
    return (tp + tn) / (tp + tn + fp + fn)


def recall(tp: int, fn: int) -> float:
    """
    Calculate recall.

    Parameters:
        tp (int): True positives.
        fn (int): False negatives.

    Returns:
        float: Recall value.
    """
    return tp / (tp + fn)


def precision(tp: int, fp: int) -> float:
    """
    Calculate precision.

    Parameters:
        tp (int): True positives.
        fp (int): False positives.

    Returns:
        float: Precision value.
    """
    return tp / (tp + fp)


def negative_predictive_value(tn: int, fn: int) -> float:
    """
    Calculate negative predictive value.

    Parameters:
        tn (int): True negatives.
        fn (int): False negatives.

    Returns:
        float: Negative predictive value.
    """
    return tn / (tn + fn)


def f1(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate F1 score.

    Parameters:
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: F1 score.
    """
    p = precision(tn, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.

        Parameters:
            obj: Object to be serialized.

        Returns:
            JSON-serializable object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_json_file(file_path) -> Dict:
    """
    Load JSON data from a file.

    Parameters:
        file_path (Path): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def num_tokens_from_string(string: str, model: str) -> int:
    """
    Returns the number of tokens in a text string.

    Parameters:
        string (str): Input text string.
        model (str): Model name.

    Returns:
        int: Number of tokens.
    """
    if not hasattr(num_tokens_from_string, "cached_encodings"):
        num_tokens_from_string.cached_encodings = {}
    if model not in num_tokens_from_string.cached_encodings:
        num_tokens_from_string.cached_encodings[model] = tiktoken.encoding_for_model(
            model
        )
    encoding = num_tokens_from_string.cached_encodings[model]
    num_tokens = len(encoding.encode(string))
    return num_tokens


def rename_datasets(df: pd.DataFrame, preserve_sampled=True) -> pd.DataFrame:
    """
    Rename datasets in a DataFrame for clarity.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        preserve_sampled (bool, optional): Preserve 'Sampled' suffix in dataset names.

    Returns:
        pd.DataFrame: DataFrame with renamed datasets.
    """
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
    """
    Calculate classification metrics including precision, recall, F1-score, and accuracy.

    Parameters:
        truths (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Model predictions.

    Returns:
        Tuple[float, float, float, float]: Precision, Recall, F1-score, and Accuracy.
    """
    prec = precision_score(truths, predictions)
    rec = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    return prec, rec, f1, acc
