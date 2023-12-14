from pathlib import Path
import pandas as pd
import seaborn as sns

from erllm import EVAL_FOLDER_PATH, FIGURE_FOLDER_PATH


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
    g = sns.relplot(
        data=df,
        x=relation["x"],
        y=relation["y"],
        hue="measure",
        col="dataset",
        kind="line",
        col_wrap=3,
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_axis_labels(relation["xlabel"], relation["ylabel"])
    g.savefig(save_to / f"{relname}.png")


if __name__ == "__main__":
    selected_sims = [
        "jaccard",
        "overlap",
        # "mongeelkan",
        # "genjaccard",
        "cosine_sim",
        # "euclidean_sim",
    ]
    relations = {
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
            "xlabel": "Coverage",
            "ylabel": "Max. Possible Recall",
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

    save_to = FIGURE_FOLDER_PATH / "discarder"
    save_to.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(f"{EVAL_FOLDER_PATH}/discarder_stats.csv")
    df = df[df["measure"].isin(selected_sims)]
    for dataset, dataset_subset in df.groupby("dataset"):
        for rname, r in relations.items():
            plot_ds(dataset_subset, dataset, rname, r, save_to)
    for rname, r in relations.items():
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
