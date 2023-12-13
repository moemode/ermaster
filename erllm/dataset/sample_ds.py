from pathlib import Path
import pandas as pd
from erllm import DATASET_FOLDER_PATH, ORIGINAL_DATASET_NAMES
from erllm.dataset.load_ds import load_into_df


def sample(dataset: str, folder: Path, N: int) -> pd.DataFrame:
    """
    Sample N elements from the specified dataset, preserving the label ratio.
    If the dataset has less than N elements, all elements are returned.

    Parameters:
    - dataset (str): Name of the dataset.
    - folder (Path): Path to the dataset folder.
    - N (int): Number of elements to sample.

    Returns:
    - pd.DataFrame: Sampled DataFrame.

    The sampled DataFrame is saved in a new folder named "dataset_N".
    """
    fnames = ["test.csv", "train.csv", "valid.csv"]
    dataset = folder.parts[-1]
    new_folder = folder.parent / (dataset + f"_{N}")
    new_folder.mkdir(parents=True, exist_ok=True)
    df = load_into_df([folder / fname for fname in fnames if (folder / fname).exists()])
    matches = df[df["label"] == 1]
    non_matches = df[df["label"] == 0]
    if len(matches) + len(non_matches) < N:
        sample_matches, sample_non_matches = matches, non_matches
    else:
        # Sample from each group while preserving the ratio
        sample_matches = matches.sample(int(N * len(matches) / len(df)), replace=False)
        sample_non_matches = non_matches.sample(N - len(sample_matches), replace=False)
    sampled_df = pd.concat([sample_matches, sample_non_matches])
    sampled_df.to_csv(new_folder / "test.csv", index=True, index_label="_id")
    return sampled_df


if __name__ == "__main__":
    datasets = ORIGINAL_DATASET_NAMES
    root_folder = DATASET_FOLDER_PATH
    for dataset, folder in [(dataset, root_folder / dataset) for dataset in datasets]:
        if (root_folder / dataset).exists():
            sample(dataset, folder, 1250)
