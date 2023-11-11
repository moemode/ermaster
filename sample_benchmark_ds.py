from pathlib import Path
import pandas as pd
from load_benchmark import load_into_df
from preclassify import DATASET_NAMES


def sample(dataset, folder, N):
    # sample N elements from df preserving the ratio of matches where df["label"]=1 to non-matches where df["label"]=0
    fnames = ["test.csv", "train.csv", "valid.csv"]
    dataset = folder.parts[-1]
    new_folder = folder.parent / (dataset + f"_{N}")
    new_folder.mkdir(parents=True, exist_ok=True)
    df = load_into_df([folder / fname for fname in fnames if (folder / fname).exists()])
    # Separate the dataset into two DataFrames based on the label
    matches = df[df["label"] == 1]
    non_matches = df[df["label"] == 0]
    if len(matches) + len(non_matches) < N:
        sample_matches, sample_non_matches = matches, non_matches
    else:
        # Sample from each group while preserving the ratio
        sample_matches = matches.sample(int(N * len(matches) / len(df)), replace=False)
        sample_non_matches = non_matches.sample(N - len(sample_matches), replace=False)
    # Concatenate the samples
    sampled_df = pd.concat([sample_matches, sample_non_matches])
    sampled_df.to_csv(new_folder / "test.csv", index=True, index_label="_id")
    return sampled_df


if __name__ == "__main__":
    datasets = DATASET_NAMES
    root_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets"
    )
    for dataset, folder in [(dataset, root_folder / dataset) for dataset in datasets]:
        sample(dataset, folder, 1250)
