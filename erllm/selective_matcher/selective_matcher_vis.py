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
from erllm.utils import rename_datasets


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
    # Create the relplot for the specified dataset
    g = sns.relplot(
        x="threshold",
        y="Value",
        hue="Metric",
        kind="line",
        marker="o",
        data=data,
    )
    g.set_axis_labels("threshold", "Value")
    g.fig.subplots_adjust(top=0.92)
    # Set title for the plot
    g.fig.suptitle(f"{dataset_name}")
    g.savefig(save_to / f"selective_{postfix}-{dataset_name}.png")


CONFIGURATIONS = {
    "all-metrics": {
        "selected_metrics": ["precision", "recall", "f1"],
        "postfix": "all",
    },
    "no-cost": {
        "selected_metrics": ["precision", "recall", "f1"],
        "postfix": "nc",
    },
    "f1-cost": {
        "selected_metrics": ["f1"],
        "postfix": "fc",
    },
}

FIGURE_SAVE_TO = FIGURE_FOLDER_PATH / "selective_matcher"
INPATH = EVAL_FOLDER_PATH / "selective_matcher" / "35_base.csv"

if __name__ == "__main__":
    for cfg in CONFIGURATIONS.keys():
        selected_metrics = CONFIGURATIONS[cfg]["selected_metrics"]
        postfix = CONFIGURATIONS[cfg]["postfix"]
        setup_plt()
        df = pd.read_csv(INPATH)
        id_vars = ["Dataset", "coverage", "threshold"]
        # Filter the dataframe to only contain the selected metrics and id_vars
        df = df[selected_metrics + id_vars]
        # Reshape the dataframe for seaborn
        df_melted = pd.melt(df, id_vars=id_vars, var_name="Metric", value_name="Value")
        df_melted = rename_datasets(df_melted, preserve_sampled=False)
        df_melted["excluded"] = 1 - df_melted["coverage"]
        # Create the relplot
        g = sns.relplot(
            x="excluded",
            y="Value",
            hue="Metric",
            col="Dataset",
            col_wrap=3,
            kind="line",
            data=df_melted,
            marker="o",
            facet_kws={"sharex": False, "sharey": False},
        )

        for item, ax in g.axes_dict.items():
            ax.set_title(item)
        # Adjust layout
        # g.fig.subplots_adjust(top=0.9)
        # g.fig.suptitle("Metrics vs Threshold for Different Datasets", fontsize=16)
        p = Path(FIGURE_SAVE_TO / f"selective_{postfix}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(p)
        # Plot each dataset
        for dataset_name, group_data in df_melted.groupby("Dataset"):
            plot_dataset(group_data, dataset_name, postfix, FIGURE_SAVE_TO)
