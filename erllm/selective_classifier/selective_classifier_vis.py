"""
This script generates performance comparison plots for the discarding matcher.
It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots
for each dataset with different configurations, such as all metrics, no cost, and F1 with cost.
It also creates plots with showing the performance on all datasets at once.
"""
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from erllm import EVAL_FOLDER_PATH, FIGURE_FOLDER_PATH
from erllm.utils import my_setup_plt, rename_datasets


def setup_plt():
    """
    Set up matplotlib styling to use Seaborn and adjust font sizes for better readability.
    """
    # Override matplotlib default styling.
    plt.style.use("seaborn-v0_8")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def plot_dataset(data: pd.DataFrame, dataset_name: str, postfix: str, save_to: Path):
    """
    Plot the performance metrics for a specific dataset and save the figure.

    Parameters:
        data (pd.DataFrame): DataFrame containing performance metrics.
        dataset_name (str): Name of the dataset.
        postfix (str): Postfix to append to the saved figure file name.
    """
    x = "coverage"
    # Create the relplot for the specified dataset
    g = sns.relplot(
        x=x,
        y="Value",
        hue="Metric",
        kind="line",
        marker="o",
        data=data,
    )
    g.set_axis_labels(x, "Value")
    g.fig.subplots_adjust(top=0.92)
    # Set title for the plot
    g.fig.suptitle(f"{dataset_name}")
    g.savefig(save_to / f"selective_{postfix}-{dataset_name}.png")


def plot_multiple_ds(df: pd.DataFrame, save_to: Path):
    n_datasets = df["Dataset"].nunique()
    col_wrap = 2 if n_datasets <= 4 else 3
    share_x = True
    # Create the relplot
    g = sns.relplot(
        x="coverage",
        y="Value",
        hue="Metric",
        col="Dataset",
        col_wrap=col_wrap,
        kind="line",
        data=df,
        # marker="o",
        facet_kws={"sharex": share_x, "sharey": True},
    )
    # Adjust layout
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle("Metrics vs Threshold for Different Datasets", fontsize=16)
    if share_x:
        # if share_x is True, then all subplots share the same x-axis -> invert only once
        g.axes[0].invert_xaxis()
    for item, ax in g.axes_dict.items():
        ax.set_title(item)
        if not share_x:
            # if share_x is False, then each subplot has its own x-axis
            ax.invert_xaxis()
    # make words in legend start with capital letter
    legend = g._legend
    for t in legend.texts:
        t.set_text(t.get_text().capitalize())
    g.savefig(save_to)


PLOT_METRICS = {
    "all": ["precision", "recall", "f1"],
    "f1": ["f1"],
}

CONFIGURATIONS = {
    "base": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": FIGURE_FOLDER_PATH / "selective_classifier" / "base",
    },
    "base-cov-selected": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base_covs.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": FIGURE_FOLDER_PATH / "selective_classifier" / "base-cov-selected",
        "datasets": ["Dblp-Scholar", "Walmart-Amazon"],
    },
    "base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "35_base_covs.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": FIGURE_FOLDER_PATH / "selective_classifier" / "base-cov",
    },
    "gpt-4-base-cov": {
        "inpath": EVAL_FOLDER_PATH / "selective_classifier" / "4_base_covs.csv",
        "plot_cfgs": PLOT_METRICS,
        "save_to": FIGURE_FOLDER_PATH / "selective_classifier" / "gpt-4-base-cov",
    },
}


if __name__ == "__main__":
    for cfg_name, cfg in CONFIGURATIONS.items():
        cfg["save_to"].mkdir(parents=True, exist_ok=True)
        for postfix, selected_metrics in cfg["plot_cfgs"].items():
            my_setup_plt()
            df = pd.read_csv(cfg["inpath"])
            id_vars = ["Dataset", "coverage", "threshold"]
            # Filter the dataframe to only contain the selected metrics and id_vars
            df = df[selected_metrics + id_vars]
            # Reshape the dataframe for seaborn
            df_melted = pd.melt(
                df, id_vars=id_vars, var_name="Metric", value_name="Value"
            )
            df_melted = rename_datasets(df_melted, preserve_sampled=False)
            df_melted["excluded"] = 1 - df_melted["coverage"]
            if "datasets" in cfg:
                df_melted = df_melted[df_melted["Dataset"].isin(cfg["datasets"])]
            p = cfg["save_to"] / f"selective_{postfix}.png"
            plot_multiple_ds(df_melted, p)
            # Plot each dataset
            for dataset_name, group_data in df_melted.groupby("Dataset"):
                plot_dataset(group_data, dataset_name, postfix, cfg["save_to"])
