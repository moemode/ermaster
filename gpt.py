import os
import openai
from pathlib import Path
import tiktoken
import json
from tqdm import tqdm
from prompt_creation import prompts_targets
from typing import Dict, Optional
from utils import retry_with_exponential_backoff, numbered_path

model = "gpt-3.5-turbo-instruct"
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


def write_json_iter(it, N, fh):
    print("[", file=fh)
    for n, rec in tqdm(enumerate(it), total=N):
        if n > 0:
            print(",", file=fh)
        json.dump(rec, fh)
    print("\n]", file=fh)


def get_completions(prompts, targets, model_params):
    for p, t in zip(prompts, targets):
        resp = completions_with_backoff(prompt=p, **model_params)
        yield {"p": p, "t": t, "c": resp.choices[0].to_dict_recursive()}


def num_tokens_from_string(string, model):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def batch_get_completions(prompts, targets, model, model_params):
    batch_prompts = []
    batch_truths = []
    batch_tokens = 0
    for p, t in zip(prompts, targets):
        pt = num_tokens_from_string(p, model)
        if batch_tokens + pt + len(batch_prompts) * model_params["max_tokens"] < 3800:
            batch_prompts.append(p)
            batch_truths.append(t)
            batch_tokens += pt
        else:
            r = completions_with_backoff(prompt=batch_prompts, **model_params)
            for choice in r.choices:
                yield {
                    "p": batch_prompts[choice.index],
                    "t": batch_truths[choice.index],
                    "c": choice,
                }
            batch_prompts = [p]
            batch_truths = [t]
            batch_tokens = pt
    if len(batch_prompts) > 0:
        r = completions_with_backoff(prompt=batch_prompts, **model_params)
        for choice in r.choices:
            yield {
                "p": batch_prompts[choice.index],
                "t": batch_truths[choice.index],
                "c": choice,
            }


def fitting_prefix(start, prompts, max_completion_token):
    batch_tokens = 0
    end = start
    while True:
        if end >= len(prompts):
            return end
        new_tokens = num_tokens_from_string(prompts[end], model)
        if batch_tokens + (end - start + 1) * max_completion_token + new_tokens >= 3800:
            return end
        end += 1
        batch_tokens += new_tokens


def batch_get_completions(prompts, truths, model, model_params):
    batch_prompts = []
    batch_truths = []
    start = 0
    end = fitting_prefix(start, prompts, model_params["max_tokens"])
    while start < end:
        batch_prompts = prompts[start:end]
        batch_truths = truths[start:end]
        r = completions_with_backoff(prompt=batch_prompts, **model_params)
        for choice in r.choices:
            yield {
                "p": batch_prompts[choice.index],
                "t": batch_truths[choice.index],
                "c": choice,
            }
        start = end
        end = fitting_prefix(start, prompts, model_params["max_tokens"])


"""
dataset = "0_beer"
model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
prompts, targets = prompts_targets(Path(f"data/{dataset}.json"))
task = dataset
completions = []
run_path = Path(f"runs/{task}_{model}.json")
if run_path.exists():
    assert False


openai.api_key = os.getenv("OPENAI_API_KEY")
with open(f"runs/{task}_{model}.json", "w+") as f:
    write_json_iter(
        batch_get_completions(prompts, targets, model, model_params), len(prompts), f
    )
    # write_json_iter(get_completions(prompts[:2], targets[:2], model_params), 2, f)
"""


def run_test(dataset: Path, model_params: Dict, description: Optional[str] = None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset_name = dataset.stem
    prompts, targets = prompts_targets(dataset)
    model = model_params["model"]
    run_path = numbered_path(Path(f"runs/{dataset_name}_{model}_{description}.json"))
    with open(run_path, "w+") as f:
        write_json_iter(
            batch_get_completions(prompts, targets, model, model_params),
            len(prompts),
            f,
        )


if __name__ == "__main__":
    model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
    run_test(
        Path("/home/v/coding/ermaster/data/0_beer.json"), model_params, "1max_token"
    )
