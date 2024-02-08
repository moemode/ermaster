import json
import os
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Dict, Iterable, Optional
import openai

from erllm import PROMPTS_FOLDER_PATH, RUNS_FOLDER_PATH, SAMPLED_DATASET_NAMES
from erllm.llm_matcher.gpt_chat import get_chat_completions
from erllm.llm_matcher.prompts import Prompt, prompt_dict_to_prompts
from erllm.utils import (
    num_tokens_from_string,
    numbered_path,
    retry_with_exponential_backoff,
    write_json_iter,
)

PRICE_PER_1K_TOKENS_PROMPT = 0.002
PRICE_PER_1K_TOKENS_COMPLETE = 0.002


def completions_with_backoff(**kwargs) -> tuple[float, openai.Completion]:
    """
    This function receives a prompt and options and reqeusts an OpenAI completion.
    It also measures the time taken for the API call.
    Parameters:
    **kwargs (dict): Keyword arguments to be passed to `openai.Completion.create`.
    Returns:
    Tuple[float, openai.Completion]: A tuple containing the time taken for the API call
    and the OpenAI completion object.
    """
    start = timer()
    c = openai.Completion.create(**kwargs)
    return timer() - start, c


# adds exponential backoff to completions_with_backoff, to handle API rate limits
completions_with_backoff = retry_with_exponential_backoff(
    completions_with_backoff,
    initial_delay=60,
    exponential_base=1.1,
    max_retries=3,
    jitter=False,
)


def get_completions(prompts: list[Prompt], model_params: Dict) -> Iterable[dict]:
    """
    Generate OpenAI completions for a list of prompts.
    Makes one call to the OpenAI API for each prompt.

    This function takes a list of prompts, requests OpenAI completions for each prompt,
    and yields a dictionary containing relevant information for each completion.

    Parameters:
    prompts (list[Prompt]): A list of Prompt objects containing prompt information.
    model_params (Dict): Model parameters to be passed to `completions_with_backoff`.

    Yields:
    dict: A dictionary containing information about the completion for each prompt,
          including IDs, prompt text, truth, completion details, time spent, and token counts.
    """
    for p in prompts:
        time_spent, r = completions_with_backoff(prompt=p.prompt_string, **model_params)
        yield {
            "id0": p.id0,
            "id1": p.id1,
            "p": p.prompt_string,
            "t": p.truth,
            "c": r.choices[0],
            "d": time_spent,
            "i": r["usage"]["prompt_tokens"],
            "o": r["usage"]["completion_tokens"],
        }


def fitting_prefix(start: int, prompts: list[Prompt], model_params: Dict):
    """
    This function determines the end index within a list of prompts s.t.
    prompts[start:end] fits within the context size of a 4k model.
    It is a helper for get_completion_batch.

    Parameters:
    start (int): The starting index for the batch.
    prompts (list[Prompt]): A list of Prompt objects containing prompt information.
    model_params (Dict): Model parameters including 'model' and 'max_tokens'.

    Returns:
    int: The fitting end index for the batch of prompts.
    """
    batch_tokens = 0
    end = start
    while True:
        if end >= len(prompts):
            return end
        new_tokens = num_tokens_from_string(
            prompts[end].prompt_string, model_params["model"]
        )
        if (
            batch_tokens + (end - start + 1) * model_params["max_tokens"] + new_tokens
            >= 3800
        ):
            return end
        if model_params["max_tokens"] > 1 and (end - start + 1) > 20:
            return end
        end += 1
        batch_tokens += new_tokens


def get_completions_batch(prompts: list[Prompt], model_params: Dict) -> Iterable[Dict]:
    """
    Generate OpenAI completions by splitting prompts into batches.
    Makes one call to the OpenAI API for each batch.

    This function processes prompts in batches, making API calls for each batch, and
    yields dictionaries containing information for each completion choice.

    Parameters:
    prompts (list[Prompt]): A list of Prompt objects containing prompt information.
    model_params: Model parameters to be passed to `completions_with_backoff`.

    Yields:
    dict: A dictionary for each completion choice, including IDs, prompt text, truth,
          completion details, and time spent.
    """
    batch_prompts = []
    start = 0
    end = fitting_prefix(start, prompts, model_params)
    total_tokens = 0
    while start < end:
        batch_prompts = prompts[start:end]
        time_spent, r = completions_with_backoff(
            prompt=list(map(lambda p: p.prompt_string, batch_prompts)), **model_params
        )
        total_tokens += r["usage"]["total_tokens"]
        for choice in r.choices:
            prompt: Prompt = batch_prompts[choice.index]
            yield {
                "id0": prompt.id0,
                "id1": prompt.id1,
                "p": prompt.prompt_string,
                "t": prompt.truth,
                "c": choice,
                "d": time_spent,
                "i": r["usage"]["prompt_tokens"],
                "o": r["usage"]["completion_tokens"],
                "tt": r["usage"]["total_tokens"],
            }
        start = end
        end = fitting_prefix(start, prompts, model_params)
    print(total_tokens)


def run_test(
    prompt_file: Path,
    model_params: Dict,
    description: Optional[str] = None,
    save_to_folder: Path = RUNS_FOLDER_PATH,
    completion_function: Callable[
        [list[Prompt], Dict], Iterable[Dict]
    ] = get_completions_batch,
):
    """
    Run a test using prompts from a file and save the results.

    This function reads prompts from a file, processes them using the specified model
    parameters, and saves the results to a numbered file in the specified folder.

    Parameters:
    prompt_file (Path): Path to the file containing prompts in JSON format.
    model_params (Dict): Model parameters to be passed to `get_completions_batch`.
    description (Optional[str]): A description to include in the saved file name.
    save_to_folder (Path): Folder path to save the results.
    """
    openai.api_key = os.getenv("OAIST_KEY")
    with open(prompt_file, "r") as f:
        prompt_dict = json.load(f)
    prompts = list(prompt_dict_to_prompts(prompt_dict))
    dataset = prompt_dict["dataset"]
    model = model_params["model"]
    modelstr = model.replace("-", "_")
    run_path = numbered_path(
        Path(f"{save_to_folder}/{prompt_file.stem}-{modelstr}-{description}.json")
    )
    with open(run_path, "w") as f:
        write_json_iter(completion_function(prompts, model_params), f, len(prompts))


CONFIGURATIONS = {
    "gpt35-on-base": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH / f"{d}-general_complex_force.json",
            SAMPLED_DATASET_NAMES,
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "1max_token",
        "save_to_folder": RUNS_FOLDER_PATH / "35_base",
    },
    "gpt35-on-base-wattr-names": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH
            / "wattr_names"
            / f"{d}-general_complex_force.json",
            SAMPLED_DATASET_NAMES,
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "wattr_names",
        "save_to_folder": RUNS_FOLDER_PATH / "35_base" / "wattr_names",
    },
    "gpt35-on-base-wattr-names-rnd-order": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH
            / "wattr_names_rnd_order"
            / f"{d}-general_complex_force.json",
            filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES),
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "wattr_names",
        "save_to_folder": RUNS_FOLDER_PATH / "35_base" / "wattr_names_rnd_order",
    },
    "gpt35-on-base-wattr-names-embed-05": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH
            / "wattr_names_embed_05"
            / f"{d}-general_complex_force.json",
            filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES),
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "wattr_names_embed_05",
        "save_to_folder": RUNS_FOLDER_PATH / "35_base" / "wattr_names_embed_05",
    },
    "gpt35-on-base-wattr-names-embed-one-ppair": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH
            / "wattr_names_embed_one_ppair"
            / f"{d}-general_complex_force.json",
            filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES),
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "wattr_names_embed_one_ppair",
        "save_to_folder": RUNS_FOLDER_PATH / "35_base" / "wattr_names_embed_one_ppair",
    },
    "gpt35-on-hash": {
        "completions_function": get_completions_batch,
        "model": "gpt-3.5-turbo-instruct",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH / f"{d}-general_complex_force_hash.json",
            SAMPLED_DATASET_NAMES,
        ),
        "model_params": dict(
            model="gpt-3.5-turbo-instruct",
            max_tokens=1,
            logprobs=5,
            temperature=0,
            seed=0,
        ),
        "description": "1max_token",
        "save_to_folder": RUNS_FOLDER_PATH / "35_hash",
    },
    "gpt4-on-base": {
        "completions_function": get_chat_completions,
        "model": "gpt-4-0613",
        "prompt_paths": map(
            lambda d: PROMPTS_FOLDER_PATH / f"{d}-general_complex_force.json",
            SAMPLED_DATASET_NAMES,
        ),
        "model_params": dict(
            model="gpt-4-0613",
            max_tokens=1,
            top_logprobs=5,
            logprobs=True,
            temperature=0,
            seed=0,
        ),
        "description": "1max_token",
        "save_to_folder": RUNS_FOLDER_PATH / "4_base",
    },
}


if __name__ == "__main__":
    cfg = CONFIGURATIONS["gpt35-on-base-wattr-names-embed-one-ppair"]
    for p in cfg["prompt_paths"]:
        run_test(
            p,
            cfg["model_params"],
            cfg["description"],
            cfg["save_to_folder"],
            cfg["completions_function"],
        )
