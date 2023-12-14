import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
from erllm import PROMPT_DATA_FOLDER_PATH, PROMPTS_FOLDER_PATH, SAMPLED_DATASET_NAMES


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


def prompt_dict(
    prompt_type: str,
    prompt_data: Iterable[Dict],
    entity_type: str = "",
    entity_type_plural: str = "",
    postfix: str = "",
) -> Dict:
    """
    Convert serialized entity pairs in prompt data to a dictionary representing full prompts.
    Args:
        prompt_type (str): Type of the prompt.
        prompt_data (Iterable[Dict]): Iterable containing serialized entity pairs.
        entity_type (str, optional): Type of entity. Defaults to "".
        entity_type_plural (str, optional): Plural form of entity type. Defaults to "".
        postfix (str, optional): Additional string to append. Defaults to "".

    Returns:
        Dict: A dictionary with the following keys:
            - "prefix": The prompt prefix, depending on the prompt_type, is shared between all prompts.
            - "pairs": A list of dictionaries representing each serialized entity pair.
                Each dictionary contains:
                    - "id0": ID of the first entity.
                    - "id1": ID of the second entity.
                    - "p": Prompt string for the entity pair which is appended to prefix.
                    - "t": Truth label for the entity pair.

    The function formats the prompt prefix based on the prompt_type and creates a list of dictionaries,
    """
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
    save_to: Path = PROMPTS_FOLDER_PATH,
) -> None:
    """
    Convert serialized entity pairs from a JSON file to a JSON file containing full prompts.
    The function reads serialized entity pairs from a JSON file specified by 'prompt_data_fp'.
    It converts the serialized data into a dictionary representing full prompts using the 'prompt_dict' function.
    The resulting dictionary is saved to a JSON file with a name based on the dataset and prompt type.

    Raises:
        ValueError: If prompt type is domain but entity type or entity type plural is empty.
    """
    if prompt_type.startswith("domain") and (
        entity_type == "" or entity_type_plural == ""
    ):
        raise ValueError(
            "Prompt type is domain but entity type or entity type plural is empty."
        )
    with open(prompt_data_fp, "r") as f:
        data = json.load(f)
        d = prompt_dict(prompt_type, data, entity_type, entity_type_plural, postfix)
        # add dataset name to prompt dict
        d["dataset"] = prompt_data_fp.stem
        outpath = (
            save_to / f"{prompt_data_fp.stem}-{prompt_type}{prompt_data_fp.suffix}"
        )
        with open(outpath, "w") as f:
            json.dump(d, f, indent=2)


def prompt_dict_to_prompts(prompt_dict: Dict) -> Iterable[Prompt]:
    """
    Convert a dictionary representing full prompts to a list of Prompt objects
    by concatenating the prefix with the prompt string for each entity pair.
    """
    prefix = prompt_dict["prefix"]
    for pair in prompt_dict["pairs"]:
        yield Prompt(prefix + pair["p"], int(pair["id0"]), int(pair["id1"]), pair["t"])


CONFIGURATIONS = {
    "base": {
        "datasets": [
            Path(PROMPT_DATA_FOLDER_PATH / f"{dataset}.json")
            for dataset in SAMPLED_DATASET_NAMES
        ],
        "prompt_type": "general_complex_force",
        "postfix": "",
        "save_to": PROMPTS_FOLDER_PATH,
    },
    "hash": {
        "datasets": [
            Path(PROMPT_DATA_FOLDER_PATH / f"{dataset}.json")
            for dataset in SAMPLED_DATASET_NAMES
        ],
        "prompt_type": "general_complex_force_hash",
        "postfix": "####",
        "save_to": PROMPTS_FOLDER_PATH,
    },
}

if __name__ == "__main__":
    cfg = CONFIGURATIONS["base"]
    cfg["save_to"].mkdir(parents=True, exist_ok=True)
    for dataset in cfg["datasets"]:
        prompt_data_to_prompt_dict(
            dataset, cfg["prompt_type"], postfix=cfg["postfix"], save_to=cfg["save_to"]
        )
