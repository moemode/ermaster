from pathlib import Path
import pandas as pd
import seaborn as sns
import math
from erllm import DATASET_NAMES, EVAL_FOLDER_PATH, FIGURE_FOLDER_PATH
from erllm.utils import rename_datasets, setup_plt


def plot_ds(
    df: pd.DataFrame, dataset: str, relname: str, relation: dict, save_to: Path
):
    """
    Generate and save a line plot for a specific dataset and relation.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        dataset (str): The name of the dataset.
        relname (str): The name of the relation to be plotted.
        relation (dict): A dictionary specifying the x and y axes, as well as labels for the plot.

    Returns:
        None: The function saves the generated plot as a PNG file in the save_to directory.
    """
    g = sns.relplot(
        data=df, x=relation["x"], y=relation["y"], hue="measure", kind="line"
    )
    g.fig.subplots_adjust(top=0.925)
    g.set_axis_labels(relation["xlabel"], relation["ylabel"])
    g.fig.suptitle(f"{dataset}")
    g.savefig(save_to / f"{relname}-{dataset}.png")


def plot_all_ds(df: pd.DataFrame, relname: str, relation: dict, save_to: Path):
    """
    Generate and save a line plot for all datasets based on a specific relation.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        relname (str): The name of the relation to be plotted.
        relation (dict): A dictionary specifying the x and y axes, as well as labels for the plot.

    Returns:
        None: The function saves the generated plot as a PNG file in the save_to directory.
    """

    num_datasets = len(df["dataset"].unique())
    col_wrap = int(math.sqrt(num_datasets))

    g = sns.relplot(
        data=df,
        x=relation["x"],
        y=relation["y"],
        hue="measure",
        col="dataset",
        kind="line",
        col_wrap=col_wrap,
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(relation["xlabel"], relation["ylabel"])
    g.savefig(save_to / f"{relname}.png")


RELATIONS = {
    "n_fn": {
        "x": "n_discarded",
        "y": "n_false_negatives",
        "xlabel": "N",
        "ylabel": "FN",
    },
    "coverage_risk": {
        "x": "coverage",
        "y": "risk",
        "xlabel": "Coverage",
        "ylabel": "Risk",
    },
    "coverage_fnr": {
        "x": "coverage",
        "y": "fnr",
        "xlabel": "Coverage",
        "ylabel": "FNR",
    },
    "coverage_max_tpr": {
        "x": "coverage",
        "y": "max_tpr",
        "xlabel": "RR",
        "ylabel": "PC",
    },
}
measure_relation = {
    "mval_max_tpr": {
        "x": "measure_value",
        "y": "max_tpr",
        "xlabel": "",
        "ylabel": "Max. Possible Recall",
    },
}


CONFIGURATIONS = {
    "joc_on_nine": {
        "selected_sims": ["jaccard", "overlap", "cosine_sim"],
        "datasets": DATASET_NAMES,
        "save_to": FIGURE_FOLDER_PATH / "discarder",
    },
    "joc_on_characteristic": {
        "selected_sims": ["jaccard", "overlap", "cosine_sim"],
        "datasets": [
            "structured_walmart_amazon",
            "structured_fodors_zagats",
            "textual_abt_buy",
            "dbpedia10k",
        ],
        "save_to": FIGURE_FOLDER_PATH / "discarder" / "characteristic",
    },
}

if __name__ == "__main__":
    setup_plt()
    # sns.set_theme()
    # sns.set_context("paper")
    cfg = CONFIGURATIONS["joc_on_characteristic"]
    save_to = cfg["save_to"]
    save_to.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(f"{EVAL_FOLDER_PATH}/discarder_stats.csv")
    df = df[df["dataset"].isin(cfg["datasets"])]
    df = df[df["measure"].isin(cfg["selected_sims"])]
    df = rename_datasets(df)
    for dataset, dataset_subset in df.groupby("dataset"):
        for rname, r in RELATIONS.items():
            plot_ds(dataset_subset, dataset, rname, r, save_to)
    for rname, r in RELATIONS.items():
        plot_all_ds(df, rname, r, save_to)
    """
    # takes long time ~ 10 minutes
    for measure in ["overlap"]:
        df_measure = df[df["measure"] == measure]
        plot_all_ds(
            df_measure, f"{measure}_max_tpr", measure_relation["mval_max_tpr"], save_to
        )
    """
    print(df)
