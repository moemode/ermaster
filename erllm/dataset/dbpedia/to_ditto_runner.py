if __name__ == "__main__":
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
        to_ditto(entities, ditto_file)
