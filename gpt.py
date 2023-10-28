import os
import openai
from pathlib import Path
import tiktoken
import json
from tqdm import tqdm
from prompt_creation import prompts_targets

from utils import retry_with_exponential_backoff

model = "gpt-3.5-turbo-instruct"
PRICE_PER_1K_TOKENS_PROMPT = 0.002
PRICE_PER_1K_TOKENS_COMPLETE = 0.002


def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


completions_with_backoff = retry_with_exponential_backoff(
    completions_with_backoff, initial_delay=60, exponential_base=1.1, max_retries=3
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
        c = completions_with_backoff(prompt=p, **model_params)
        yield {"p": p, "t": t, "c": c.to_dict_recursive()}


def num_tokens_from_string(string, model):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


dataset = "0_beer20.json"
model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
prompts, targets = prompts_targets(Path(f"data/{dataset}"))
task = dataset
completions = []
run_path = Path(f"runs/{task}_{model}.json", "w")
if run_path.exists():
    assert False


openai.api_key = os.getenv("OPENAI_API_KEY")
with open(f"runs/{task}_{model}.json", "w+") as f:
    write_json_iter(get_completions(prompts[:6], targets[:6], model_params), 6, f)
