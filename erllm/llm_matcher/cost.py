"""
Provides cost calculations for language models based on specified configurations, 
including input and output costs.
"""

from collections import namedtuple
from pathlib import Path
from typing import Dict, List
import pandas as pd
from erllm import EVAL_FOLDER_PATH, PROMPTS_FOLDER_PATH, SAMPLED_DATASET_NAMES
from erllm.utils import load_json_file, num_tokens_from_string, rename_datasets

# Define a named tuple for input and output costs
ModelCost = namedtuple("ModelCost", ["input", "output"])

# maps model names to ModelCosts
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


def prompt_tokens(prompt_file: Path, model: str) -> tuple[int, int]:
    """
    Calculate the number of prompts and total tokens for a given prompt file and model.

    Args:
        prompt_file (Path): Path to the JSON file containing prompts.
        model (str): Name of the language model.

    Returns:
        tuple[int, int]: Number of prompts and total tokens.
    """
    promptsJson = load_json_file(prompt_file)
    pairs = promptsJson["pairs"]
    N = len(pairs)
    entity_tokens = sum(num_tokens_from_string(pair["p"], model) for pair in pairs)
    prefix_tokens = num_tokens_from_string(promptsJson["prefix"], model)
    return N, N * prefix_tokens + entity_tokens


def cost(
    prompt_file: Path,
    outtokens_per_prompt: int,
    model: str,
    costs: Dict[str, ModelCost] = MODEL_COSTS,
) -> tuple[float, float]:
    """
    Calculate the cost of generating output tokens for a given prompt file and model.

    Args:
        prompt_file (Path): Path to the JSON file containing prompts.
        outtokens_per_prompt (int): Number of output tokens per prompt.
        model (str): Name of the language model.
        costs (Dict[str, ModelCost], optional): Dictionary mapping model names to cost values.

    Returns:
        tuple[float, float]: A tuple containing the total cost of generating output tokens and the cost per prompt.

    Raises:
        ValueError: If the specified model does not have associated costs.
    """
    if model not in costs:
        raise ValueError(f"No costs available for {model}")
    N, intokens = prompt_tokens(prompt_file, model)
    outtokens = N * outtokens_per_prompt
    total_cost = (
        intokens / 1000 * costs[model].input + outtokens / 1000 * costs[model].output
    )
    return total_cost, total_cost / N


def str_cost(
    prompt_strings: List[str],
    outtokens_per_prompt: int,
    model: str,
    costs: Dict[str, ModelCost] = MODEL_COSTS,
) -> float:
    """
    Calculate the cost of generating output tokens for a given list of prompt strings and model.

    Args:
        prompt_strings (List[str]): List of prompt strings.
        outtokens_per_prompt (int): Number of output tokens per prompt.
        model (str): Name of the language model.
        costs (Dict[str, ModelCost], optional): Dictionary mapping model names to cost values.

    Returns:
        float: Cost of generating output tokens.
    """
    N = len(prompt_strings)
    # Iterate over pairs and calculate f, then sum the results
    intokens = sum(num_tokens_from_string(p, model) for p in prompt_strings)
    outtokens = N * outtokens_per_prompt
    return intokens / 1000 * costs[model].input + outtokens / 1000 * costs[model].output


CONFIGURATIONS = {
    "base": {
        "models": ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview"],
        "prompt_paths": map(
            lambda d: (d, PROMPTS_FOLDER_PATH / f"{d}-general_complex_force.json"),
            SAMPLED_DATASET_NAMES,
        ),
        "outtokens_per_prompt": 1,
    },
    "hash": {
        "models": ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview"],
        "prompt_paths": map(
            lambda d: (d, PROMPTS_FOLDER_PATH / f"{d}-general_complex_force.json"),
            SAMPLED_DATASET_NAMES,
        ),
        "outtokens_per_prompt": 10,
    },
}


if __name__ == "__main__":
    cfg = CONFIGURATIONS["base"]
    results = []
    for dataset, dataset_path in cfg["prompt_paths"]:
        for model in cfg["models"]:
            dataset_cost, cost_per_pair = cost(
                dataset_path, cfg["outtokens_per_prompt"], model
            )
            results.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Cost": dataset_cost,
                    "Cost per Pair": cost_per_pair,
                }
            )
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    df = rename_datasets(df, False)
    # get results where model = "gpt-3.5-turbo-instruct"
    df = df[df["Model"] == "gpt-3.5-turbo-instruct"]
    # leave out model column
    df = df.drop(columns=["Model"])

    s = df.style
    s.format(subset=["Cost"], formatter="{:,.2f}\$", escape="latex")
    s.format(subset=["Cost per Pair"], formatter="{:,.5f}\$", escape="latex")
    s.hide(axis="index")
    # generate latex table, save to file
    s.to_latex(EVAL_FOLDER_PATH / "llm_matcher" / "costs.ltx", hrules=True)
    print(s.to_latex())
