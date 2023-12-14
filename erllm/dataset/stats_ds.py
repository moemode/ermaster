from pathlib import Path
import pandas as pd
from erllm import DATASET_FOLDER_PATH, DATASET_NAMES, EVAL_WRITEUP_FOLDER_PATH
from erllm.dataset.load_ds import load_dataset


def dataset_statistics(
    folder: Path,
) -> tuple[int, int, int]:
    """
    Compute dataset statistics.

    Parameters:
    - folder (Path): The folder containing the dataset.

    Returns:
    tuple[int, int, int]: The number of pairs, the number of positive pairs, and the number of negative pairs.
    """
    save_to = Path("prompt_data")
    save_to.mkdir(parents=True, exist_ok=True)
    pairs = load_dataset(folder, use_tqdm=True)
    n_pos = len(list(filter(lambda p: p[0] == True, pairs)))
    n_neg = len(list(filter(lambda p: p[0] == False, pairs)))
    return len(pairs), n_pos, n_neg


if __name__ == "__main__":
    datasets = DATASET_NAMES
    data = []
    root_folder = DATASET_FOLDER_PATH
    for folder in [root_folder / dataset for dataset in datasets]:
        dataset = folder.parts[-1]
        data.append((dataset, *dataset_statistics(folder)))
    # create df from data
    df_columns = ["Dataset", "Total Pairs", "Positive Pairs", "Negative Pairs"]
    df = pd.DataFrame(data, columns=df_columns)
    # Print or further process the DataFrame
    print(df)
    df.to_csv(f"{EVAL_WRITEUP_FOLDER_PATH}/dataset_statistics.csv", index=False)
