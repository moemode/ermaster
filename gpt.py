import json
import os
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Optional

import openai
from preclassify import SAMPLED_DATASET_NAMES

from prompts import Prompt, prompt_dict_to_prompts
from utils import (
    num_tokens_from_string,
    numbered_path,
    retry_with_exponential_backoff,
    write_json_iter,
)

PRICE_PER_1K_TOKENS_PROMPT = 0.002
PRICE_PER_1K_TOKENS_COMPLETE = 0.002


def completions_with_backoff(**kwargs):
    start = timer()
    c = openai.Completion.create(**kwargs)
    return timer() - start, c


completions_with_backoff = retry_with_exponential_backoff(
    completions_with_backoff,
    initial_delay=60,
    exponential_base=1.1,
    max_retries=3,
    jitter=False,
)


def get_completions(prompts, targets, model_params):
    for p, t in zip(prompts, targets):
        resp = completions_with_backoff(prompt=p, **model_params)
        yield {"p": p, "t": t, "c": resp.choices[0].to_dict_recursive()}


def fitting_prefix(start: int, prompts: list[Prompt], model_params: Dict):
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


def get_completions(prompts: list[Prompt], model_params):
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


def get_completions_batch(prompts: list[Prompt], model_params):
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
            }
        start = end
        end = fitting_prefix(start, prompts, model_params)
    print(total_tokens)


def run_test(
    prompt_file: Path,
    model_params: Dict,
    description: Optional[str] = None,
):
    openai.api_key = os.getenv("OAIST_KEY")
    with open(prompt_file, "r") as f:
        prompt_dict = json.load(f)
    prompts = list(prompt_dict_to_prompts(prompt_dict))
    dataset = prompt_dict["dataset"]
    model = model_params["model"]
    modelstr = model.replace("-", "_")
    run_path = numbered_path(
        Path(f"runs/{prompt_file.stem}-{modelstr}-{description}.json")
    )
    with open(run_path, "w") as f:
        write_json_iter(get_completions_batch(prompts, model_params), f, len(prompts))


if __name__ == "__main__":
    model = "gpt-3.5-turbo-instruct"
    datasets = SAMPLED_DATASET_NAMES
    # remove dbpedia, and textual_company
    # datasets = filter(lambda ds: "textual_company" not in ds, datasets)
    datasets = ["dbpedia10k_1250"]
    # datasets = filter(lambda ds: "dblp_acm" in ds, datasets)
    model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0, seed=0)
    paths = map(
        lambda d: Path("prompts") / f"{d}-general_complex_force.json",
        datasets,
    )
    paths = list(filter(lambda p: p.exists(), paths))
    print(list(paths))
    for p in paths:
        run_test(
            p,
            model_params,
            "1max_token",
        )
    """
    run_test(
        Path(
            "/home/v/coding/ermaster/prompts/dbpedia10k-2_1250-general_complex_force.json"
        ),
        model_params,
        "1max_token",
    )
    """
