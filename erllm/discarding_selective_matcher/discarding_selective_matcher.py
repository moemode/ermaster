from pathlib import Path
from typing import Any, Dict, Set, Tuple
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from erllm import RUNS_FOLDER_PATH, SIMILARITIES_FOLDER_PATH
from erllm.discarding_matcher.discarding_matcher import find_matching_csv
from erllm.llm_matcher.cost import str_cost
from erllm.llm_matcher.evalrun import CompletedPrompt, read_run_raw


def discarding_selective_matcher(
    discard_fraction: float,
    label_fraction: float,
    runFile: Path,
    similaritiesFile: Path,
    sim_function="overlap",
) -> Tuple[
    Dict[Tuple[int, int], bool],
    Set[Tuple[int, int]],
    Set[Tuple[int, int]],
    Set[Tuple[int, int]],
    float,
    float,
]:
    """
    Runs the discarding selective matcher and returns its predictions and intermediate data like the identifiers of discarded pairs.

    Args:
        discard_fraction (float): The fraction of dataset pairs to discard.
        label_fraction (float): The fraction of dataset pairs to label.
        runFile (Path): The path to the run file containing completions for dataset pairs.
        similaritiesFile (Path): The path to the file containing pairwise similarities between dataset pairs.
        sim_function (str, optional): The similarity function to use. Defaults to "overlap".

    Returns:
        Tuple[Dict[Tuple[int, int], bool], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], float, float]:
            - predictions (Dict[Tuple[int, int], bool]): A dictionary mapping dataset pair IDs to their predicted labels.
            - discarded_pairs (Set[Tuple[int, int]]): A set of dataset pairs that were discarded.
            - llm_pairs (Set[Tuple[int, int]]): A set of dataset pairs that were labeled by the selective matcher.
            - human_label_pairs (Set[Tuple[int, int]]): A set of dataset pairs that were labeled by human experts.
            - discarder_threshold (float): The threshold value used by the discarding algorithm.
            - confidence_threshold (float): The threshold value used by the selective matcher.
    """
    if discard_fraction + label_fraction > 1:
        raise ValueError(
            "The sum of discard_fraction and label_fraction must not be greater than 1."
        )
    completions = read_run_raw(runFile)
    dataset_pair_ids: Set[Tuple[int, int]] = set(completions.keys())
    sims = get_similarities(dataset_pair_ids, similaritiesFile)
    n_discard = int(len(dataset_pair_ids) * discard_fraction)
    discarded, kept = discarder(sims, n_discard, sim_function)
    discarder_threshold = discarded[sim_function].max()
    discarded_pairs = set(zip(discarded["table1.id"], discarded["table2.id"]))
    kept_pairs = set(zip(kept["table1.id"], kept["table2.id"]))
    kept_completions = {
        pair_id: cp for pair_id, cp in completions.items() if pair_id in kept_pairs
    }
    n_label = int(len(dataset_pair_ids) * label_fraction)
    predictions, llm_pairs, human_label_pairs, confidence_threshold = selective_matcher(
        n_label, kept_completions
    )
    predictions.update({pair_id: False for pair_id in discarded_pairs})
    assert len(predictions) == len(dataset_pair_ids)
    assert len(discarded_pairs) + len(llm_pairs) == len(dataset_pair_ids)
    return (
        predictions,
        discarded_pairs,
        llm_pairs,
        human_label_pairs,
        discarder_threshold,
        confidence_threshold,
    )


def get_similarities(
    dataset_ids: set[Tuple[int, int]],
    similarities_file: Path,
) -> pd.DataFrame:
    """
    Retrieves the similarities between pairs of dataset IDs from a CSV file.

    Args:
        dataset_ids (set[Tuple[int, int]]): A set of tuples representing pairs of dataset IDs.
        similarities_file (Path): The path to the CSV file containing the similarities.

    Returns:
        pd.DataFrame: A DataFrame containing the similarities between the specified dataset ID pairs.

    Raises:
        ValueError: If the similarity file does not contain values for some pairs or if the number of rows in the similarity file does not match the expected number of pairs.
    """
    similarities = pd.read_csv(similarities_file)
    sim_pairs = set(zip(similarities["table1.id"], similarities["table2.id"]))
    if not set(dataset_ids).issubset(sim_pairs):
        raise ValueError("Similarity file misses values for some pairs.")
    similarities = similarities[
        similarities.apply(
            lambda row: (row["table1.id"], row["table2.id"]) in dataset_ids, axis=1
        )
    ]
    if not len(similarities) == len(dataset_ids):
        raise ValueError(
            f"Similarity file has {len(similarities)} rows, but {len(dataset_ids)} pairs were expected."
        )
    return similarities


def discarder(
    similarities: pd.DataFrame, n_discard: int, sim_function="overlap"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discards a specified number of rows from a DataFrame based on a similarity function.

    Args:
        similarities (pd.DataFrame): The DataFrame containing the similarity values.
        n_discard (int): The number of rows to discard.
        sim_function (str, optional): The name of the similarity function column. Defaults to "overlap".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: the discarded rows and the kept rows.
    """
    if not sim_function in similarities.columns:
        raise ValueError(
            f"Similarity file does not have a column named {sim_function}."
        )
    similarities_sorted = similarities.sort_values(by=[sim_function])
    discarded_rows = similarities_sorted.head(n_discard)
    kept_rows = similarities_sorted.tail(len(similarities) - n_discard)
    return discarded_rows, kept_rows


def selective_matcher(
    n_label: int, completions: dict[Tuple[int, int], CompletedPrompt]
) -> Tuple[
    dict[Tuple[int, int], bool], Set[Tuple[int, int]], Set[Tuple[int, int]], float
]:
    """
    Returns predictions based on completions but sets the n_label least confident predictions to the true label.
    This simulates perfect manual labeling.

    Args:
        n_label (int): The number of labels to match.
        completions (dict[Tuple[int, int], CompletedPrompt]): A dictionary of completions.

    Returns:
        Tuple[dict[Tuple[int, int], bool], Set[Tuple[int, int]], Set[Tuple[int, int]], float]:
            - predictions: A dictionary of label predictions.
            - llm_ids: A set of all label IDs.
            - least_confident_ids: A set of the least confident label IDs.
            - threshold: The confidence score threshold.

    """
    predictions: dict[Tuple[int, int], bool] = {
        id_pair: cp.prediction for id_pair, cp in completions.items()
    }
    # get the n_label id pairs with smallest confidence
    sorted_completions = sorted(completions.values(), key=lambda cp: cp.probability)
    least_confident_ids = set((cp.id0, cp.id1) for cp in sorted_completions[:n_label])
    threshold = sorted_completions[n_label - 1].probability if n_label > 0 else 0
    llm_ids = set(completions.keys())
    for id_pair in least_confident_ids:
        predictions[id_pair] = completions[id_pair].truth
    return predictions, llm_ids, least_confident_ids, threshold


def llm_cost_duration(
    pairs: Set[tuple[int, int]],
    completions: dict[Tuple[int, int], CompletedPrompt],
    outtokens_per_prompt: int,
    model: str,
) -> tuple[float, float]:
    """
    Calculate the total cost and duration for obtaining the completions.
    Args:
        pairs (Set[tuple[int, int]]): A set of pairs representing the indices of prompts.
        completions (dict[Tuple[int, int], CompletedPrompt]): A dictionary mapping pairs to CompletedPrompt objects.
        outtokens_per_prompt (int): The number of output tokens per prompt.
        model (str): The name of the language model.

    Returns:
        tuple[float, float]: A tuple containing the cost and duration of running the language model matcher.
    """
    prompts = [completions[p].prompt_string for p in pairs]
    cost = str_cost(prompts, outtokens_per_prompt, model)
    duration = sum([completions[p].duration for p in pairs])
    return cost, duration


def discarder_duration(
    pairs: Set[tuple[int, int]], simfile: Path, sim_function: str
) -> float:
    """
    Calculate the duration the discarder alone takes on deciding on the pairs.

    Args:
        pairs (Set[tuple[int, int]]): A set of pairs representing table IDs.
        simfile (Path): The path to the similarity file.
        sim_function (str): The name of the similarity function.

    Returns:
        float: The total duration of the discarding selective matcher.
    """
    similarities = pd.read_csv(simfile)
    sims_for_pairs = similarities[
        similarities.apply(
            lambda row: (row["table1.id"], row["table2.id"]) in pairs, axis=1
        )
    ]
    assert len(sims_for_pairs) == len(pairs)
    duration = sims_for_pairs[sim_function + "_dur"].sum()
    return duration


def intermediate_stats(
    completions: Dict[tuple[int, int], CompletedPrompt],
    discarded_pairs: Set[tuple[int, int]],
    llm_pairs: Set[tuple[int, int]],
    human_label_pairs: Set[tuple[int, int]],
    simfile: Path,
    sim_function: str,
) -> dict[str, float | int]:
    """
    Calculate intermediate statistics for the discarding selective matcher based on the output of
    discarding_selective_matcher function.

    Args:
        completions (Dict[tuple[int, int], CompletedPrompt]): A dictionary of completion pairs.
        discarded_pairs (Set[tuple[int, int]]): A set of discarded completion pairs.
        llm_pairs (Set[tuple[int, int]]): A set of LLM completion pairs.
        human_label_pairs (Set[tuple[int, int]]): A set of completion pairs with human labels.
        simfile (Path): The path to the similarity file.
        sim_function (str): The similarity function to use.

    Returns:
        dict: A dictionary containing the intermediate statistics:
            - "Discarded": The number of discarded pairs.
            - "Discarded FN": The number of discarded pairs with a true truth value.
            - "Discarder Duration": The duration of the discarder.
            - "LLM All Cost": The cost of LLM for all pairs.
            - "LLM All Duration": The duration of LLM for all pairs.
            - "LLM Cost": The cost of LLM for LLM pairs.
            - "LLM Duration": The duration of LLM for LLM pairs.
            - "Manual": The number of pairs with human labels.
    """
    all_pairs = set(completions.keys())
    n_discarded = len(discarded_pairs)
    discarder_dur = discarder_duration(discarded_pairs, simfile, sim_function)
    discarded_fn = [completions[p].truth for p in discarded_pairs].count(True)
    llm_all_cost, llm_all_duration = llm_cost_duration(
        all_pairs, completions, 1, "gpt-3.5-turbo-instruct"
    )
    llm_cost, llm_duration = llm_cost_duration(
        llm_pairs, completions, 1, "gpt-3.5-turbo-instruct"
    )
    n_manual = len(human_label_pairs)
    return {
        "Discarded": n_discarded,
        "Discarded FN": discarded_fn,
        "Discarder Duration": discarder_dur,
        "LLM All Cost": llm_all_cost,
        "LLM All Duration": llm_all_duration,
        "LLM Cost": llm_cost,
        "LLM Duration": llm_duration,
        "Manual": n_manual,
    }


def eval_discarding_selective_matcher(
    discard_fraction: float,
    label_fraction: float,
    runfile: Path,
    simfile: Path,
    sim_function="overlap",
) -> Dict[str, Any]:
    """
    Fully evaluate the discarding selective matcher.

    Args:
        discard_fraction (float): The fraction of pairs to discard.
        label_fraction (float): The fraction of pairs to label.
        runfile (Path): The path to the runfile.
        simfile (Path): The path to the similarity file.
        sim_function (str, optional): The similarity function to use. Defaults to "overlap".

    Returns:
        dict: A dictionary containing the evaluation metrics.
            - "N" (int): The total number of completions.
            - "Discarder Threshold" (float): The discarder threshold.
            - "Confidence Threshold" (float): The confidence threshold.
            - "Accuracy" (float): The accuracy score.
            - "Precision" (float): The precision score.
            - "Recall" (float): The recall score.
            - "F1" (float): The F1 score.
    """
    completions = read_run_raw(runfile)
    (
        predictions,
        discarded_pairs,
        llm_pairs,
        human_label_pairs,
        disc_threshold,
        conf_threshold,
    ) = discarding_selective_matcher(
        discard_fraction, label_fraction, runfile, simfile, sim_function
    )
    intermediate = intermediate_stats(
        completions,
        discarded_pairs,
        llm_pairs,
        human_label_pairs,
        simfile,
        sim_function,
    )
    # align predictions with truths to compute classification metrics
    predictions_list = []
    truths_list = []
    for id_pair, cp in completions.items():
        predictions_list.append(predictions[id_pair])
        truths_list.append(cp.truth)
    return {
        **intermediate,
        "N": len(completions),
        "Discarder Threshold": disc_threshold,
        "Confidence Threshold": conf_threshold,
        "Accuracy": accuracy_score(truths_list, predictions_list),
        "Precision": precision_score(truths_list, predictions_list),
        "Recall": recall_score(truths_list, predictions_list),
        "F1": f1_score(truths_list, predictions_list),
    }


if __name__ == "__main__":
    # example run for debugging
    CONFIGURATIONS = {
        "base": {
            "runfiles": RUNS_FOLDER_PATH / "35_base",
            "similarities": SIMILARITIES_FOLDER_PATH,
        },
    }
    for path in CONFIGURATIONS["base"]["runfiles"].glob(
        "*dblp_scholar*force-gpt*.json"
    ):
        dataset_name = path.stem.split("-")[0]
        simPath = find_matching_csv(
            path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
        )
        if not simPath:
            raise ValueError(
                f"No matching similarity file in {CONFIGURATIONS['base']['similarities']} found for {path}"
            )
        print(eval_discarding_selective_matcher(0.3, 0.1, path, simPath))
