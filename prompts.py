import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from preclassify import SAMPLED_DATASET_NAMES


@dataclass
class Prompt:
    prompt_string: str
    id0: int
    id1: int
    truth: bool


VERB_CONFIDENCE = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: "

DOMAIN_SIMPLE = "Do the two {entity_type_plural} match?"
DOMAIN_COMPLEX = "Do the two {entity_type_plural} refer to the same real-world product?"
GENERAL_SIMPLE = "Do the two entity descriptions match?"
GENERAL_COMPLEX = "Do the two entity descriptions refer to the same real-world entity?"
FORCE = "Answer with 'Yes' if they do and 'No' if they do not."
DOMAIN_PAIR = "{entity_type} 1: '{e0}'\n{entity_type} 2: '{e1}'\n"
GENERAL_PAIR = "Entity 1: '{e0}'\nEntity 2: '{e1}'\n"
FORCE_HASH = "Answer with ####Yes if they do and ####No if they do not."


TASK_PREFIXES = {
    "domain_simple_free": f"{DOMAIN_SIMPLE}\n",
    "domain_complex_free": f"{DOMAIN_COMPLEX}\n",
    "domain_simple_force": f"{DOMAIN_SIMPLE} {FORCE}\n",
    "domain_complex_force": f"{DOMAIN_COMPLEX} {FORCE}\n",
    "general_simple_free": f"{GENERAL_SIMPLE}\n",
    "general_complex_free": f"{GENERAL_COMPLEX}\n",
    "general_simple_force": f"{GENERAL_SIMPLE} {FORCE}\n",
    "general_complex_force": f"{GENERAL_COMPLEX} {FORCE}\n",
    "general_complex_force_hash": f"{GENERAL_COMPLEX} {FORCE_HASH}\n",
}

newitems = []
for name, prompt_prefix in TASK_PREFIXES.items():
    newitems.append(("guess_" + name, VERB_CONFIDENCE + prompt_prefix))
for k, v in newitems:
    TASK_PREFIXES[k] = v


SIMPLE_PROMPT_POSTFIX = """Do the two entity descriptions match?
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer with 'Yes' if they do and 'No' if they do not.
"""


def prompt_dict(
    prompt_type: str,
    prompt_data: Iterable[Dict],
    entity_type: str = "",
    entity_type_plural: str = "",
    postfix: str = "",
) -> Dict:
    is_domain = prompt_type.startswith("domain")
    prefix = TASK_PREFIXES[prompt_type].format(
        entity_type_plural=entity_type_plural,
    )
    pair_str = DOMAIN_PAIR if is_domain else GENERAL_PAIR
    pairs = []
    for pair in prompt_data:
        pairs.append(
            {
                "id0": pair["id0"],
                "id1": pair["id1"],
                "p": pair_str.format(**pair, entity_type=entity_type) + postfix,
                "t": pair["t"],
            }
        )
    return {"prefix": prefix, "pairs": pairs}


def prompt_data_to_prompt_dict(
    prompt_data_fp: Path,
    prompt_type: str,
    entity_type: str = "",
    entity_type_plural: str = "",
    postfix: str = "",
) -> Path:
    if prompt_type.startswith("domain") and (
        entity_type == "" or entity_type_plural == ""
    ):
        raise ValueError(
            "Prompt type is domain but entity type or entity type plural is empty."
        )
    promptFolder = Path("prompts")
    promptFolder.mkdir(parents=True, exist_ok=True)
    with open(prompt_data_fp, "r") as f:
        data = json.load(f)
        d = prompt_dict(prompt_type, data, entity_type, entity_type_plural, postfix)
        d["dataset"] = prompt_data_fp.stem
        outpath = (
            promptFolder / f"{prompt_data_fp.stem}-{prompt_type}{prompt_data_fp.suffix}"
        )
        with open(outpath, "w") as f:
            json.dump(d, f, indent=2)
    return outpath


def prompt_dict_to_prompts(prompt_dict: Dict) -> Iterable[Prompt]:
    prefix = prompt_dict["prefix"]
    for pair in prompt_dict["pairs"]:
        yield Prompt(prefix + pair["p"], int(pair["id0"]), int(pair["id1"]), pair["t"])


if __name__ == "__main__":
    for dataset in SAMPLED_DATASET_NAMES:
        prompt_data_to_prompt_dict(
            Path("prompt_data") / f"{dataset}.json",
            "general_complex_force_hash",
            postfix="####",
        )

"""
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
"""
