import pandas as pd
from pathlib import Path
from erllm import DATASET_FOLDER_PATH, SAMPLED_DATASET_NAMES
from erllm.dataset.dbpedia.access_dbpedia import entities_from_dbpedia_df
from erllm.dataset.ditto.to_ditto import ditto_split, to_ditto_task
from erllm.dataset.load_ds import load_dataset

cfg = {
    "dataset_paths": [
        DATASET_FOLDER_PATH / dataset
        for dataset in filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES)
    ],
}

if __name__ == "__main__":
    label_fraction = 0.15
    valid_train_ratio = 0.25
    dbpedia1250_csv = DATASET_FOLDER_PATH / "dbpedia10k_1250/test.csv"
    dataset_paths = [
        DATASET_FOLDER_PATH / dataset
        for dataset in filter(lambda x: "dbpedia" not in x, SAMPLED_DATASET_NAMES)
    ]
    ditto_task_folder = Path("/home/v/coding/ermaster/data/benchmark_datasets/ditto")
    dsp_labeled_pairs = []
    for dsp in dataset_paths:
        labeled_pairs = load_dataset(dsp)
        dsp_labeled_pairs.append((dsp, labeled_pairs))
    df = pd.read_csv(dbpedia1250_csv)
    labeled_pairs = entities_from_dbpedia_df(df)
    dsp_labeled_pairs.append((dbpedia1250_csv.parent, labeled_pairs))
    for dsp, labeled_pairs in dsp_labeled_pairs:
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
    old way to generate DBPedia ditto dataset
    label_fraction = 0.15
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
        pairs_to_ditto(entities, ditto_file)
    """
