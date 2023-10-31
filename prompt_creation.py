import json
from pathlib import Path
from typing import Dict, List
from utils import write_json_iter

SIMPLE_PROMPT = """Do the two entity descriptions match? Answer with 'Yes' if they do and 'No' if they do not.
Entity 1: '{e0}'
Entity 2: '{e1}'
"""

SIMPLE_PROMPT_POSTFIX = """Do the two entity descriptions match?
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer with 'Yes' if they do and 'No' if they do not.
"""


def simple_postfix(element: Dict) -> str:
    return SIMPLE_PROMPT_POSTFIX.format(**element)


def simple(element: Dict) -> str:
    return SIMPLE_PROMPT.format(**element)


def create_prompts(file: Path, prompt_function=simple_postfix) -> List[str]:
    p = []
    try:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for element in data:
                formatted_prompt = prompt_function(element)
                print(formatted_prompt)
                p.append(formatted_prompt)
        return p
    except FileNotFoundError:
        print(f"File not found: {file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_prompts_iter(file: Path, prompt_function=simple_postfix):
    with open(file, "r") as json_file:
        data = json.load(json_file)
        for element in data:
            yield {"p": prompt_function(element), "t": element["t"]}


def save_prompts(file: Path, prompt_function=simple_postfix):
    out_path = Path(f"prompts/{file.stem}_{prompt_function.__name__}{file.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w+") as pf:
        write_json_iter(create_prompts_iter(file, prompt_function), None, pf)


def prompts_targets(file: Path, pattern=SIMPLE_PROMPT_POSTFIX) -> List[str]:
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


def get_targets(file: Path) -> List[str]:
    t = []
    try:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for element in data:
                t.append(element["t"])
        return t
    except FileNotFoundError:
        print(f"File not found: {file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
