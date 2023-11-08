from pathlib import Path
from access_dbpedia import OrderedEntity
from load_benchmark import load_benchmark
from py_stringmatching import (
    Jaccard,
    OverlapCoefficient,
    MongeElkan,
    GeneralizedJaccard,
)
import pandas as pd
import matplotlib.pyplot as plt


def similarities(
    pairs: list[tuple[bool, OrderedEntity, OrderedEntity]]
) -> pd.DataFrame:
    """
    Calculate the similarity between pairs of entities and return the results as a pandas DataFrame.

    Args:
        pairs (List[Tuple[bool, OrderedEntity, OrderedEntity]]): A list of tuples, each containing three elements:
            1. A boolean value representing the label for the entity pair.
            2. An OrderedEntity object representing the first entity.
            3. An OrderedEntity object representing the second entity.

    Returns:
        pd.DataFrame: A DataFrame containing similarity scores for each entity pair with the following format:

        |  table1.id  |  table2.id  |  label  |  jaccard  |  overlap  |  mongeelkan  |  genjaccard  |
        |------------|------------|--------|----------|----------|-------------|-------------|
        |    3505    |    30346   |   0    |  0.280000|  0.538462|   0.767434  |   0.544994  |
        |    5343    |    47894   |   0    |  0.477273|  0.700000|   0.877984  |   0.620952  |
        |    ...     |    ...     |  ...   |   ...    |   ...    |    ...      |    ...      |

    The function calculates the similarity between pairs of entities. It assumes that the first entity in each pair
    is the entity from the first dataset, and the second entity in each pair is the entity from the second dataset.
    """
    jac = Jaccard()
    overlap = OverlapCoefficient()
    me = MongeElkan()
    gj = GeneralizedJaccard()
    similarity_scores = []
    for label, e0, e1 in pairs:
        t0 = e0.tokens()
        t1 = e1.tokens()
        # Calculate similarity scores using different similarity functions
        jac_score = jac.get_raw_score(t0, t1)
        overlap_score = overlap.get_raw_score(t0, t1)
        me_score = me.get_raw_score(t0, t1)
        gj_score = gj.get_raw_score(t0, t1)
        # Append the similarity scores to the result list
        similarity_scores.append(
            (e0.id, e1.id, label, jac_score, overlap_score, me_score, gj_score)
        )
    columns = [
        "table1.id",
        "table2.id",
        "label",
        "jaccard",
        "overlap",
        "mongeelkan",
        "genjaccard",
    ]
    df = pd.DataFrame(similarity_scores, columns=columns)
    return df


def count_miss_classifications(similarities: pd.DataFrame, name: str):
    # Sort the DataFrame by the specified column in ascending order
    similarities.sort_values(by=name, ascending=True, inplace=True)
    match_count = 0
    miss_classifiation_function = []
    for i, (_, row) in enumerate(similarities.iterrows()):
        if row["label"] == 1:
            match_count += 1
        miss_classifiation_function.append((i + 1, match_count))
    return miss_classifiation_function


if __name__ == "__main__":
    fnames = ["test.csv", "train.csv", "valid.csv"]
    folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets/structured_itunes_amazon"
    )
    # print(get_folders_in_directory(BENCHMARKS_PATH))
    pairs = load_benchmark([folder / fname for fname in fnames])
    s = similarities(pairs)
    data = count_miss_classifications(s, "jaccard")

    # Extract the x and y values from the data for plotting
    x_values, y_values = zip(*data)

    # Create a graph using Matplotlib
    plt.plot(x_values, y_values, label="Miss Classifications")
    plt.xlabel("Row Index")
    plt.ylabel("Count of Miss Classifications")
    plt.title("Miss Classifications vs. Row Index")
    plt.legend()
    plt.grid(True)

    # Display the graph
    plt.show()
