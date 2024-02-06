from pathlib import Path
from erllm import DATASET_FOLDER_PATH, SAMPLED_DATASET_NAMES
from erllm.dataset.load_ds import load_dataset
from erllm.dataset.to_ditto import to_ditto_task, ditto_split

cfg = {
    "dataset_paths": [
        DATASET_FOLDER_PATH / dataset
        for dataset in filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES)
    ],
}

if __name__ == "__main__":
    label_fraction = 0.15
    valid_train_ratio = 0.25
    dataset_paths = [
        DATASET_FOLDER_PATH / dataset
        for dataset in filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES)
    ]
    ditto_task_folder = Path("/home/v/coding/ermaster/data/benchmark_datasets/ditto")
    for dsp in dataset_paths:
        labeled_pairs = load_dataset(dsp)
        train_fraction, valid_fraction = (
            label_fraction * (1 - valid_train_ratio),
            label_fraction * valid_train_ratio,
        )
        train, valid, test = ditto_split(
            labeled_pairs, train_fraction, valid_fraction, seed=123
        )
        outfolder = ditto_task_folder / dsp.parts[-1]
        to_ditto_task(train, valid, test, outfolder)
    """
    train, valid, test = 
    dbpedia1250_csv = (
        DATA_FOLDER_PATH
        / "benchmark_datasets/existingDatasets/dbpedia10k_1250/test.csv"
    )
    dbpedia10k_csv = (
        DATA_FOLDER_PATH / "benchmark_datasets/existingDatasets/dbpedia10k/train.csv"
    )
    dbpedia_ditto_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/ditto/dbpedia"
    )
    dbpedia_ditto_folder.mkdir(parents=True, exist_ok=True)
    dfs = dbpedia_to_train_valid_test(dbpedia1250_csv, dbpedia10k_csv, label_fraction)
    for df, stem in zip(dfs, ("train", "valid", "test")):
        entities = entities_from_dbpedia_df(df)
        ditto_file = dbpedia_ditto_folder / f"{stem}.txt"
        to_ditto(entities, ditto_file)
    """
