import pandas as pd

from erllm import EVAL_FOLDER_PATH, SIMILARITIES_FOLDER_PATH


def get_stat_functions(similarities: pd.DataFrame, sim_name: str):
    """Calculate statistics about discarding based on given similarity column.
    Sort the profile pairs in ascending order by the specified similarity column.
    Calculate functions for various statistic like number of false negatives dependent on the number of discarded pairs.
    Args:
        similarities (pd.DataFrame): A DataFrame with the similarity values. Has structure like this
        |  table1.id  |  table2.id  |  label  |  jaccard  |  overlap  |  mongeelkan  |  genjaccard  |
        |------------|------------|--------|----------|----------|-------------|-------------|
        |    3505    |    30346   |   0    |  0.280000|  0.538462|   0.767434  |   0.544994  |
        |    5343    |    47894   |   0    |  0.477273|  0.700000|   0.877984  |   0.620952  |
        |    ...     |    ...     |  ...   |   ...    |   ...    |    ...      |    ...      |

        sim_name (str): The name of the similarity column.

    Returns:
        A list of tuples, capturing the functions of the following values dependent on the number of discarded pairs (from 0 to all discarded):
        - measure_value: The value of the similarity measure
        - n_false_negatives: The number of false negatives
        - coverage: The coverage
        - risk: The risk
        - fnr: The false negative rate
    """
    # Sort the DataFrame by the specified column in ascending order
    similarities.sort_values(by=sim_name, ascending=True, inplace=True)
    n_total = len(similarities)
    n_positive = similarities["label"].sum()
    n_fn = 0
    min_sim = similarities[sim_name].min()
    stats = [(0, min_sim, 0, 0, 0, 0)]
    for i, (_, row) in enumerate(similarities.iterrows()):
        if row["label"] == 1:
            n_fn += 1
        coverage = (i + 1) / n_total
        risk = n_fn / (i + 1)
        fnr = n_fn / n_positive
        stats.append((i + 1, row[sim_name], n_fn, coverage, risk, fnr))
    return stats


if __name__ == "__main__":
    file_paths = SIMILARITIES_FOLDER_PATH.glob("*-allsim.csv")
    data_list = []
    for f in file_paths:
        ds = f.stem.split("-")[0]
        s = pd.read_csv(f)
        similarity_columns = [
            "jaccard",
            "overlap",
            "mongeelkan",
            "genjaccard",
            "cosine_sim",
            "euclidean_sim",
        ]
        for sim_name in similarity_columns:
            # Calculate statistics for each similarity column
            if sim_name not in s.columns:
                continue
            stats = get_stat_functions(s, sim_name)
            for n, measure_value, n_fn, coverage, risk, fnr in stats:
                data_list.append(
                    [ds, sim_name, n, measure_value, n_fn, coverage, risk, fnr]
                )
    # Create one large dataframe containing all the statistics for all similarity functions
    df = pd.DataFrame(
        data_list,
        columns=[
            "dataset",
            "measure",
            "n_discarded",
            "measure_value",
            "n_false_negatives",
            "coverage",
            "risk",
            "fnr",
        ],
    )
    # add column max_tpr = 1 - fnr
    df["max_tpr"] = 1 - df["fnr"]
    (EVAL_FOLDER_PATH / "discarder").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{EVAL_FOLDER_PATH}/discarder/stats.csv", index=False)
