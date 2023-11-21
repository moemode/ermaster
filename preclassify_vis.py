import pandas as pd
import seaborn as sns


def plot_ds(df: pd.DataFrame, dataset: str, relname: str, relation: dict):
    g = sns.relplot(
        data=df, x=relation["x"], y=relation["y"], hue="measure", kind="line"
    )
    g.set_axis_labels(relation["xlabel"], relation["ylabel"])
    # Set title
    g.fig.suptitle(f"{dataset}")
    g.savefig(f"figures/{relname}-{dataset}-missclassifications.png")


def plot_all_ds(df: pd.DataFrame, relname: str, relation: dict):
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
    # Set axis labels
    g.set_axis_labels(relation["xlabel"], relation["ylabel"])
    # Save the Seaborn plot to a file
    g.savefig(f"figures/{relname}-missclassifications.png")


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

    df = pd.read_csv("eval/missclassifications.csv")
    df = df[df["measure"].isin(selected_sims)]
    for dataset, dataset_subset in df.groupby("dataset"):
        for rname, r in relations.items():
            plot_ds(dataset_subset, dataset, rname, r)
    for rname, r in relations.items():
        plot_all_ds(df, rname, r)
    print(df)
