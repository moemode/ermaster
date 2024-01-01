"""
Provides functions to evaluate the performance of the LLM Mathcer on a set of run files produced by the OpenAI GPT API. 
It calculates various classification metrics, entropies, and calibration results. 
The evaluation results are saved as JSON files for individual runs and aggregated into a CSV file for further analysis.
"""
from pathlib import Path
from typing import Iterable
import pandas as pd
from erllm import EVAL_FOLDER_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import eval
from erllm.calibration.reliability_diagrams import *

LLM_MATCHER_FOLDER_PATH = EVAL_FOLDER_PATH / "llm_matcher"


def eval_dir(json_files: Iterable[Path], save_to: Path):
    """
    Evaluate multiple run files and save the results to a CSV file.

    Parameters:
        json_files (Iterable[Path]): Iterable of paths to JSON run files.
        fname (str): Name of the output CSV file.
    """
    all_results = []  # List to store results for each file
    for file in json_files:
        ds, prompt_type, model, description = file.parts[-1].split("-")
        save_ind_to = save_to.parent / "individual"
        save_ind_to.mkdir(parents=True, exist_ok=True)
        results = eval(file, save_ind_to)
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
    df.to_csv(save_to, index=False)
    print(df)


if __name__ == "__main__":
    LLM_MATCHER_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    CONFIGURATIONS = {
        "base": "35_base/*force-gpt*.json",
        "hash": "35_hash/*force_hash-gpt*.json",
        # move the files from 35_base and 35_hash into 35_base_hash if you want this
        "base_hash": "35_base_hash/*.json",
        "gpt4-base": "4_base/*force-gpt*.json",
    }
    for cfg in CONFIGURATIONS.keys():
        eval_dir(
            Path(RUNS_FOLDER_PATH).glob(CONFIGURATIONS[cfg]),
            save_to=EVAL_FOLDER_PATH / "llm_matcher" / f"{cfg}.csv",
        )
