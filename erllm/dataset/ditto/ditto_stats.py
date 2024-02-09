from pathlib import Path

from erllm import DATA_FOLDER_PATH


def ditto_stats(ditto_task_file: Path):
    """
    Calculate statistics for a given Ditto task file.

    Parameters:
    ditto_task_file (Path): The path to the Ditto task file.

    Returns:
    tuple: A tuple containing the number of total samples, number of positive samples, and number of negative samples.
    """
    # open file and count number of non-empty lines
    with open(ditto_task_file, "r", encoding="utf-8") as file:
        lines = tuple(filter(lambda l: len(l) != 0, file.readlines()))
    n_lines = len(lines)
    n_pos = len(tuple(filter(lambda x: x.strip().split("\t")[-1] == "1", lines)))
    return n_lines, n_pos, n_lines - n_pos


if __name__ == "__main__":
    ditto_task_file = Path(DATA_FOLDER_PATH / "dataset/ditto/dbpedia10k_1250/test.txt")
    n_lines = ditto_stats(ditto_task_file)
    print(n_lines)
