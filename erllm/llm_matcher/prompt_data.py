from typing import Callable
import json
from pathlib import Path
from erllm import DATASET_FOLDER_PATH, DATASET_NAMES, PROMPT_DATA_FOLDER_PATH
from erllm.dataset.entity import OrderedEntity
from erllm.dataset.load_ds import load_dataset


def dataset_to_prompt_data(
    folder: Path,
    save_to: Path,
    to_str: Callable[[OrderedEntity], str] = lambda e: e.value_string(),
):
    """
    Take labeled entity pairs from CSV and serialize them into string for use in prompt.
    Also keep groundtruth and ids.
    Args:
        folder (Path): Path to the CSV dataset folder.
        save_to (Path): Path to save the converted JSON data.
        to_str (Callable[[OrderedEntity], str], optional): Function to serilalize an OrderedEntity into a string.

    Converts CSV entity pairs to JSON format where
    each entry has keys "t", "id0", "id1", "e0", and "e1".
    The resulting JSON is saved to '{dataset}.json' in 'save_to'.
    Example:
    dataset_to_prompt_data(Path("path/to/dataset"), Path("path/to/save"))
    """
    dataset = folder.parts[-1]
    pairs = load_dataset(folder, use_tqdm=True)
    data = []
    for truth, e0, e1 in pairs:
        data.append(
            {"t": truth, "id0": e0.id, "id1": e1.id, "e0": to_str(e0), "e1": to_str(e1)}
        )
    with open(save_to / f"{dataset}.json", "w") as json_file:
        json.dump(data, json_file, indent=2)


CONFIGURATIONS = {
    "default": {
        "save_to": Path(PROMPT_DATA_FOLDER_PATH),
        "dataset_paths": [DATASET_FOLDER_PATH / dataset for dataset in DATASET_NAMES],
        "to_str": lambda e: e.value_string(),
    },
    "with-attr-names": {
        "save_to": Path(PROMPT_DATA_FOLDER_PATH) / "wattr_names",
        "dataset_paths": [DATASET_FOLDER_PATH / dataset for dataset in DATASET_NAMES],
        "to_str": lambda e: e.ffm_wrangle_string(),
    },
    "with-attr-names-rnd-order": {
        "save_to": Path(PROMPT_DATA_FOLDER_PATH) / "wattr_names_rnd_order",
        "dataset_paths": [
            DATASET_FOLDER_PATH / dataset
            for dataset in filter(lambda x: "dbpedia" not in x, DATASET_NAMES)
        ],
        "to_str": lambda e: e.ffm_wrangle_string(random_order=True),
    },
}

if __name__ == "__main__":
    cfg = CONFIGURATIONS["with-attr-names-rnd-order"]
    datasets, save_to, to_str = cfg["dataset_paths"], cfg["save_to"], cfg["to_str"]
    save_to.mkdir(parents=True, exist_ok=True)
    for folder in datasets:
        dataset_to_prompt_data(folder, save_to, to_str)
