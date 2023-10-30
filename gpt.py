import os
import openai
from pathlib import Path
import tiktoken
from prompt_creation import get_targets, create_prompts, save_prompts, simple
from typing import Callable, Dict, Optional
from utils import retry_with_exponential_backoff, numbered_path, write_json_iter

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


def num_tokens_from_string(string, model):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def fitting_prefix(start, prompts, model_params):
    batch_tokens = 0
    end = start
    while True:
        if end >= len(prompts):
            return end
        new_tokens = num_tokens_from_string(prompts[end], model_params["model"])
        if (
            batch_tokens + (end - start + 1) * model_params["max_tokens"] + new_tokens
            >= 3800
        ):
            return end
        if model_params["max_tokens"] > 1 and (end - start + 1) > 20:
            return end
        end += 1
        batch_tokens += new_tokens


def get_completions_batch(prompts, truths, model_params):
    batch_prompts = []
    batch_truths = []
    start = 0
    end = fitting_prefix(start, prompts, model_params)
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
        end = fitting_prefix(start, prompts, model_params)


def run_test(
    dataset: Path,
    model_params: Dict,
    prompt_function: Optional[Callable] = simple,
    description: Optional[str] = None,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset_name = dataset.stem
    prompts = create_prompts(dataset, simple)
    targets = get_targets(dataset)
    save_prompts(dataset)
    model = model_params["model"]
    run_path = numbered_path(
        Path(
            f"runs/{dataset_name}_{prompt_function.__name__}_{model}_{description}.json"
        )
    )
    with open(run_path, "w+") as f:
        write_json_iter(
            get_completions_batch(prompts, targets, model_params),
            len(prompts),
            f,
        )


if __name__ == "__main__":
    model = "gpt-3.5-turbo-instruct"
    model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
    run_test(
        Path("/home/v/coding/ermaster/data/0_beer.json"),
        model_params,
        simple,
        "1max_token",
    )
