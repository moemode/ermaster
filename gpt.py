import json
import os
from pathlib import Path
from typing import Dict, Optional

import openai

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
    return openai.Completion.create(**kwargs)


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


def get_completions_batch(prompts: list[Prompt], model_params):
    batch_prompts = []
    start = 0
    end = fitting_prefix(start, prompts, model_params)
    total_tokens = 0
    while start < end:
        batch_prompts = prompts[start:end]
        r = completions_with_backoff(
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
            }
        start = end
        end = fitting_prefix(start, prompts, model_params)
    print(total_tokens)


def run_test(
    prompt_file: Path,
    model_params: Dict,
    description: Optional[str] = None,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
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
    model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
    run_test(
        Path(
            "/home/v/coding/ermaster/prompts/dbpedia10k-2_1250_general_complex_force.json"
        ),
        model_params,
        "1max_token",
    )
