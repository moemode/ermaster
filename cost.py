from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import pandas as pd

from preclassify import DATASET_NAMES
from utils import load_json_file, num_tokens_from_string

# Define a named tuple for input and output costs
ModelCost = namedtuple("ModelCost", ["input", "output"])

# Create a dictionary mapping model names to Costs named tuples
MODEL_COSTS = {
    "gpt-4-1106-preview": ModelCost(input=0.01, output=0.03),
    "gpt-4-1106-vision-preview": ModelCost(input=0.01, output=0.03),
    "gpt-4": ModelCost(input=0.03, output=0.06),
    "gpt-4-32k": ModelCost(input=0.06, output=0.12),
    "gpt-3.5-turbo": ModelCost(input=0.0010, output=0.0020),
    "gpt-3.5-turbo-1106": ModelCost(input=0.0010, output=0.0020),
    "gpt-3.5-turbo-instruct": ModelCost(input=0.0015, output=0.0020),
    "gpt-3.5-turbo-0613": ModelCost(input=0.0015, output=0.0020),
    "gpt-3.5-turbo-16k-0613": ModelCost(input=0.0030, output=0.0040),
}

# Additional information for legacy models
legacy_model_info = {
    "gpt-3.5-turbo-0613": {
        "shutdown_date": "2024-06-13",
        "legacy_model": True,
        "legacy_model_price": ModelCost(input=0.0015, output=0.0020),
        "recommended_replacement": "gpt-3.5-turbo-1106",
    },
    "gpt-3.5-turbo-16k-0613": {
        "shutdown_date": "2024-06-13",
        "legacy_model": True,
        "legacy_model_price": ModelCost(input=0.0030, output=0.0040),
        "recommended_replacement": "gpt-3.5-turbo-1106",
    },
}


def prompt_tokens(prompt_file: Path, model: str):
    # Load JSON data from the specified file path
    promptsJson = load_json_file(prompt_file)
    pairs = promptsJson["pairs"]
    N = len(pairs)
    # Iterate over pairs and calculate f, then sum the results
    entity_tokens = sum(num_tokens_from_string(pair["p"], model) for pair in pairs)
    prefix_tokens = num_tokens_from_string(promptsJson["prefix"], model)
    return N, N * prefix_tokens + entity_tokens


def cost(
    prompt_file: Path,
    outtokens_per_prompt: int,
    model: str,
    costs: Dict[str, ModelCost] = MODEL_COSTS,
) -> float:
    if model not in costs:
        raise ValueError(f"No costs available for {model}")
    N, intokens = prompt_tokens(prompt_file, model)
    outtokens = N * outtokens_per_prompt
    return intokens / 1000 * costs[model].input + outtokens / 1000 * costs[model].output


def str_cost(
    prompt_strings: List[str],
    outtokens_per_prompt: int,
    model: str,
    costs: Dict[str, ModelCost] = MODEL_COSTS,
) -> float:
    N = len(prompt_strings)
    # Iterate over pairs and calculate f, then sum the results
    intokens = sum(num_tokens_from_string(p, model) for p in prompt_strings)
    outtokens = N * outtokens_per_prompt
    return intokens / 1000 * costs[model].input + outtokens / 1000 * costs[model].output


if __name__ == "__main__":
    paths = map(
        lambda d: (
            d,
            Path(
                f"/home/v/coding/ermaster/prompts/{d}-general_complex_force_hash.json"
            ),
        ),
        DATASET_NAMES,
    )
    paths = filter(lambda p: p[1].exists(), paths)
    models = ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview"]
    results = []
    for dataset, dataset_path in paths:
        for model in models:
            dataset_cost = cost(dataset_path, 10, model)
            results.append({"Dataset": dataset, "Model": model, "Cost": dataset_cost})
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    print(df)
