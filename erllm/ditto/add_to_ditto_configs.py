"""
Copy the datasets in DITTO format to the ditto folder into the subfolder data/erllm.
Add the new datasets to the configs.json file in the ditto folder.
"""

import json
from erllm import DATA_FOLDER_PATH
import sys
from pathlib import Path
import shutil

DITTO_FOLDER = DATA_FOLDER_PATH / "benchmark_datasets" / "ditto"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <ditto_folder>")
        sys.exit(1)
    ditto_folder = Path(sys.argv[1])
    configs_file_path = Path(sys.argv[1]) / "configs.json"
    if not configs_file_path.exists():
        print(f"configs.json not found in {sys.argv[1]}")
        sys.exit(1)
    # save a copy of the original configs.json
    shutil.copy(configs_file_path, configs_file_path.with_suffix(".json.bak"))
    ditto_configs = []
    for folder_path in DITTO_FOLDER.iterdir():
        target_folder_path = ditto_folder / "data" / "erllm" / folder_path.name
        target_folder_path.mkdir(parents=True, exist_ok=True)
        rel_target_folder_path = Path("data/erllm") / folder_path.name
        # copy folder_path to target_folder_path
        shutil.copytree(folder_path, target_folder_path, dirs_exist_ok=True)
        if folder_path.is_dir():
            entry = {
                "name": folder_path.name,
                "task_type": "classification",
                "vocab": ["0", "1"],
                "trainset": str(rel_target_folder_path / "train.txt"),
                "validset": str(rel_target_folder_path / "valid.txt"),
                "testset": str(rel_target_folder_path / "test.txt"),
            }
            # Append the entry to the Ditto configs
            ditto_configs.append(entry)

    # Append the new entries to the existing configs.json
    with configs_file_path.open("r") as configs_file:
        existing_configs = json.load(configs_file)
        existing_configs.extend(ditto_configs)

    with configs_file_path.open("w") as configs_file:
        json.dump(existing_configs, configs_file, indent=2)
    print("Entries added to configs.json")
