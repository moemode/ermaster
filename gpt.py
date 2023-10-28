import os
import openai
from pathlib import Path
import tiktoken
import json
from typing import List
from itertools import islice


def write_json_iter(it, fh):
    print("[", file=fh)
    for n, rec in enumerate(it):
        if n > 0:
            print(",", file=fh)
        json.dump(rec, fh)
    print("]", file=fh)


PROMPT = """Do the two entity descriptions match?
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer with 'Yes' if they do and 'No' if they do not.
"""


def load_data(pattern: str, file: Path) -> List[str]:
    p = []
    t = []
    try:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for element in data:
                formatted_prompt = pattern.format(**element)
                print(formatted_prompt)
                p.append(formatted_prompt)
                t.append(element["t"])
        return p, t
    except FileNotFoundError:
        print(f"File not found: {file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


model = "gpt-3.5-turbo-instruct"
PRICE_PER_1K_TOKENS_PROMPT = 0.002
PRICE_PER_1K_TOKENS_COMPLETE = 0.002


def num_tokens_from_string(string, model):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


dataset = "0_beer20.json"

# Example usage:
file_path = Path("data/{dataset}")
prompts, targets = load_data(PROMPT, file_path)
model_params = dict(model=model, max_tokens=1, logprobs=5, temperature=0)
task = dataset
completions = []
run_path = Path(f"runs/0_beer20_{model}.json", "w")
if run_path.exists():
    assert False
with open(f"runs/0_beer20_{model}.json", "w+") as f:
    print("[", file=f)
    for p, t in islice(zip(prompts, targets), 2):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        c = openai.Completion.create(prompt=p, **model_params)
        json.dump({"p": p, "t": t, "c": c.to_dict_recursive()}, f)
        print(",\n", file=f)
    print("]", file=f)
