import pandas as pd
from pathlib import Path


def get_stat_functions(similarities: pd.DataFrame, sim_name: str):
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
    # Specify the path to the CSV file
    file_paths = Path("/home/v/coding/ermaster/eval").glob("*-allsim.csv")
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
            # Calculate miss classifications for each similarity column
            # check if name exists in dataframe
            if sim_name not in s.columns:
                continue
            stats = get_stat_functions(s, sim_name)
            for n, measure_value, n_fn, coverage, risk, fnr in stats:
                data_list.append(
                    [ds, sim_name, n, measure_value, n_fn, coverage, risk, fnr]
                )
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
    df.to_csv("eval/missclassifications.csv", index=False)
