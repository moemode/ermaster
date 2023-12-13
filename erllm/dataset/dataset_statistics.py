from pathlib import Path
import pandas as pd
from erllm.dataset.load_ds import load_benchmark
from erllm.discarder.discarder import DATASET_NAMES


def dataset_statistics(
    folder: Path,
):
    save_to = Path("prompt_data")
    save_to.mkdir(parents=True, exist_ok=True)
    dataset = folder.parts[-1]
    pairs = load_benchmark(folder, use_tqdm=True)
    n_pos = len(list(filter(lambda p: p[0] == True, pairs)))
    n_neg = len(list(filter(lambda p: p[0] == False, pairs)))
    return len(pairs), n_pos, n_neg


if __name__ == "__main__":
    datasets = DATASET_NAMES
    data = []
    root_folder = Path("data/benchmark_datasets/existingDatasets")
    for folder in [root_folder / dataset for dataset in datasets]:
        dataset = folder.parts[-1]
        data.append((dataset, *dataset_statistics(folder)))
    # create df from data
    df_columns = ["Dataset", "Total Pairs", "Positive Pairs", "Negative Pairs"]
    df = pd.DataFrame(data, columns=df_columns)
    # Print or further process the DataFrame
    print(df)
    df.to_csv("eval_writeup/dataset_statistics.csv", index=False)
