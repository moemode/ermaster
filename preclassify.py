from pathlib import Path
from access_dbpedia import OrderedEntity
from py_stringmatching import (
    Jaccard,
    OverlapCoefficient,
    MongeElkan,
    GeneralizedJaccard,
)
import pandas as pd
from load_benchmark import load_benchmark
from tqdm import tqdm


DATASET_NAMES = [
    # "dirty_dblp_acm",
    # "dirty_walmart_amazon",
    # "dirty_itunes_amazon",
    # "dirty_dblp_scholar",
    "structured_dblp_acm",
    "structured_itunes_amazon",
    "textual_company",
    "structured_amazon_google",
    "structured_dblp_scholar",
    "structured_walmart_amazon",
    "structured_beer",
    "structured_fodors_zagats",
    "textual_abt_buy",
]


def similarities(
    pairs: list[tuple[bool, OrderedEntity, OrderedEntity]], use_tqdm: bool = False
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
    # Use tqdm for progress tracking
    if use_tqdm:
        pairs_iterator = tqdm(pairs, total=len(pairs))
    else:
        pairs_iterator = pairs
    for label, e0, e1 in pairs_iterator:
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


def run_all(datasets=DATASET_NAMES):
    fnames = ["test.csv", "train.csv", "valid.csv"]
    root_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets"
    )
    save_to = Path("eval")
    save_to.mkdir(parents=True, exist_ok=True)
    for folder in [root_folder / dataset for dataset in datasets]:
        dataset = folder.parts[-1]
        result_path = save_to / f"{dataset}-sim.csv"
        if result_path.exists():
            continue
        print("Load Dataset:", dataset)
        pairs = load_benchmark([folder / fname for fname in fnames], use_tqdm=True)
        print("Similiarities on Dataset:", dataset)
        s = similarities(pairs, use_tqdm=True)
        s.to_csv(result_path, index=False)


if __name__ == "__main__":
    ds = DATASET_NAMES
    # takes too long
    ds.remove("textual_company")
    run_all(ds)
