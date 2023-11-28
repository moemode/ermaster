from pathlib import Path
from typing import Iterable, Optional


def pre_llm(runFile: Path, similaritiesFile: Path):
    pass


def find_matching_csv(
    run_file: Path, similarity_files: Iterable[Path]
) -> Optional[Path]:
    """
    Find the matching similarity CSV file for a given run JSON file based on dataset name.
    Works only if the dataset name is the first part of the run JSON file name
    and contained in the similarity CSV file name.

    Parameters:
        run_file (Path): The path to the run JSON file.
        similarity_files (Iterable[Path]): Iterable of paths to similarity CSV files.

    Returns:
        Optional[Path]: The path to the matching similarity CSV file, or None if not found.
    """
    json_dataset_name = run_file.stem.split("-")[0]
    json_dataset_name = json_dataset_name.replace("_1250", "")
    matching_csv = next(
        (
            csv_file
            for csv_file in similarity_files
            if json_dataset_name in csv_file.stem
        ),
        None,
    )
    return matching_csv


if __name__ == "__main__":
    CONFIGURATIONS = {
        "base": {"runfiles": "runs/35_base/*force-gpt*.json", "similarities": "eval"},
    }
    for path in Path(".").glob(CONFIGURATIONS["base"]["runfiles"]):
        print(
            find_matching_csv(
                path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
            )
        )
