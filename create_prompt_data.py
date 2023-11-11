from typing import Callable
import json
from pathlib import Path
from access_dbpedia import OrderedEntity

from load_benchmark import load_benchmark
from preclassify import DATASET_NAMES


def dataset_to_prompt_data(
    folder: Path,
    to_str: Callable[[OrderedEntity], str] = lambda e: e.value_string(),
):
    save_to = Path("prompt_data")
    save_to.mkdir(parents=True, exist_ok=True)
    dataset = folder.parts[-1]
    pairs = load_benchmark(folder, use_tqdm=True)
    data = []
    for truth, e0, e1 in pairs:
        data.append({"t": truth, "e0": to_str(e0), "e1": to_str(e1)})
    with open(Path("prompt_data") / f"{dataset}.json", "w") as json_file:
        json.dump(data, json_file, indent=2)


if __name__ == "__main__":
    datasets = DATASET_NAMES
    sampled_datasets = [dataset + "_1250" for dataset in datasets]
    datasets.extend(sampled_datasets)
    root_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets"
    )
    for folder in [root_folder / dataset for dataset in datasets]:
        dataset_to_prompt_data(folder)
