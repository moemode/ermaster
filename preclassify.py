from pathlib import Path
import pickle
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

from sentence_transformers import SentenceTransformer
import numpy as np


ORIGINAL_DATASET_NAMES = [
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
    "dbpedia10k",
    "dbpedia10k_harder",
]
SAMPLED_DATASET_NAMES = [dataset + "_1250" for dataset in ORIGINAL_DATASET_NAMES]

DATASET_NAMES = ORIGINAL_DATASET_NAMES + SAMPLED_DATASET_NAMES


def euclidean(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))


def euclidean_sim(vector1, vector2):
    return 1 / (1 + euclidean(vector1, vector2))


def cosine_sim(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 and norm2 == 0:
        # Handle the case where both vectors have zero norm
        return np.nan
    if norm1 == 0 or norm2 == 0:
        # Handle the case where one of the vectors has zero norm
        return 0.0
    return np.dot(vector1, vector2) / (norm1 * norm2)


set_sim_functions = (
    ("jaccard", Jaccard().get_raw_score),
    ("overlap", OverlapCoefficient().get_raw_score),
    ("mongeelkan", MongeElkan().get_raw_score),
    ("genjaccard", GeneralizedJaccard().get_raw_score),
)

fast_set_sim_functions = (
    ("jaccard", Jaccard().get_raw_score),
    ("overlap", OverlapCoefficient().get_raw_score),
)


def similarities(
    pairs: list[tuple[bool, OrderedEntity, OrderedEntity]],
    similarity_functions=set_sim_functions,
    use_tqdm: bool = False,
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
    sim_names, sim_function = zip(*similarity_functions)
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
        sim_scores = (f(t0, t1) for f in sim_function)
        # Append the similarity scores to the result list
        similarity_scores.append((e0.id, e1.id, label, *sim_scores))
    columns = ["table1.id", "table2.id", "label", *sim_names]
    df = pd.DataFrame(similarity_scores, columns=columns)
    return df


def embedding_similarities(
    pairs: list[tuple[bool, OrderedEntity, OrderedEntity]],
    embeddings_path: Path,
    sim_functions=[cosine_sim, euclidean_sim],
    use_tqdm: bool = False,
) -> pd.DataFrame:
    # Load embeddings from disk
    with open(embeddings_path, "rb") as f:
        entity_embeddings = pickle.load(f)
    similarity_scores = []
    # Use tqdm for progress tracking
    if use_tqdm:
        pairs_iterator = tqdm(pairs, total=len(pairs))
    else:
        pairs_iterator = pairs
    for label, e0, e1 in pairs_iterator:
        # Retrieve embeddings for entities from the loaded dictionary
        embedding_e0 = entity_embeddings.get(e0.id)
        embedding_e1 = entity_embeddings.get(e1.id)
        if embedding_e0 is None or embedding_e1 is None:
            raise ValueError("Missing embedding")
        sims = (f(embedding_e0, embedding_e1) for f in sim_functions)
        similarity_scores.append((e0.id, e1.id, label, *sims))
    columns = [
        "table1.id",
        "table2.id",
        "label",
        *(f.__name__ for f in sim_functions),
    ]
    df = pd.DataFrame(similarity_scores, columns=columns)
    return df


def compute_embeddings(
    pairs: list[tuple[bool, OrderedEntity, OrderedEntity]],
    save_to: Path,
    use_tqdm: bool = False,
):
    model = SentenceTransformer(
        "sentence-transformers/average_word_embeddings_glove.6B.300d"
    )
    # Create a dictionary to store embeddings for each entity
    entity_embeddings = {}
    # Use tqdm for progress tracking
    if use_tqdm:
        pairs_iterator = tqdm(pairs, total=len(pairs))
    else:
        pairs_iterator = pairs
    for label, e0, e1 in pairs_iterator:
        embedding_e0 = model.encode(e0.value_string())
        embedding_e1 = model.encode(e1.value_string())
        # Store embeddings in the dictionary
        entity_embeddings[e0.id] = embedding_e0
        entity_embeddings[e1.id] = embedding_e1
    # Serialize embeddings to disk using pickle
    with open(save_to, "wb") as f:
        pickle.dump(entity_embeddings, f)


def dataset_similarities(
    dataset,
    folder,
    set_sim=set_sim_functions,
    compute_emb=True,
    emb_sim=(cosine_sim, euclidean_sim),
):
    save_to = Path("eval")
    save_to.mkdir(parents=True, exist_ok=True)
    dataset = folder.parts[-1]
    result_path_set_sim = save_to / f"{dataset}-sim.csv"
    embedding_path = folder / "embeddings.pkl"
    result_path_emb_sim = save_to / f"{dataset}-embsim.csv"
    set_sim_missing = set_sim and not result_path_set_sim.exists()
    emb_missing = compute_emb and not embedding_path.exists()
    emb_sim_missing = emb_sim and not result_path_emb_sim.exists()
    if not any([set_sim_missing, emb_missing, emb_sim_missing]):
        return
    pairs = load_benchmark(folder, use_tqdm=True)
    if set_sim_missing:
        print("Similiarities on Dataset:", dataset)
        sims = similarities(pairs, set_sim, use_tqdm=True)
        sims.to_csv(result_path_set_sim, index=False)
    if emb_missing:
        print("Embeddings on Dataset:", dataset)
        compute_embeddings(pairs, embedding_path, True)
    if emb_sim_missing:
        print("Embedding similarities on Dataset:", dataset)
        s = embedding_similarities(pairs, embedding_path, emb_sim, True)
        s.to_csv(result_path_emb_sim, index=False)
    # Combine set similarities and embedding similarities if both are computed
    result_path_all_sim = save_to / f"{dataset}-allsim.csv"
    if set_sim and emb_sim and not result_path_all_sim.exists():
        sims = pd.read_csv(result_path_set_sim)
        emb_sims = pd.read_csv(result_path_emb_sim)
        if sims is not None and emb_sims is not None:
            all_sims = pd.merge(sims, emb_sims, on=["table1.id", "table2.id", "label"])
            all_sims.to_csv(result_path_all_sim, index=False)
        else:
            print(f"Missing set similarities or embedding similarities for {dataset}")


if __name__ == "__main__":
    datasets = ORIGINAL_DATASET_NAMES
    root_folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets"
    )
    for dataset, folder in [(dataset, root_folder / dataset) for dataset in datasets]:
        set_sim = set_sim_functions
        if dataset in ["textual_company", "dbpedia10k", "dbpedia10k_harder"]:
            set_sim = fast_set_sim_functions
        dataset_similarities(dataset, folder, set_sim)
