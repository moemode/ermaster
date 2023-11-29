from pathlib import Path
import seaborn as sns
import pandas as pd

from utils import rename_datasets
import matplotlib.pyplot as plt


def setup_plt():
    # Override matplotlib default styling.
    plt.style.use("seaborn-v0_8")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def plot_dataset(data, dataset_name, postfix):
    # Create the relplot for the specified dataset
    g = sns.relplot(
        x="Threshold",
        y="Value",
        hue="Metric",
        kind="line",
        marker="o",
        data=data,
    )
    g.set_axis_labels("Threshold", "Value")
    g.fig.subplots_adjust(top=0.92)
    # Set title for the plot
    g.fig.suptitle(f"{dataset_name}")
    g.savefig(f"figures/pre_llm/pre_llm_perf_{postfix}-{dataset_name}.png")


CONFIGURATIONS = {
    "all-metrics": {
        "selected_metrics": ["Precision", "Recall", "F1", "Cost Relative"],
        "postfix": "all",
    },
    "no-cost": {
        "selected_metrics": ["Precision", "Recall", "F1"],
        "postfix": "nc",
    },
    "f1-cost": {
        "selected_metrics": ["F1", "Cost Relative"],
        "postfix": "fc",
    },
}

if __name__ == "__main__":
    for cfg in CONFIGURATIONS.keys():
        selected_metrics = CONFIGURATIONS[cfg]["selected_metrics"]
        postfix = CONFIGURATIONS[cfg]["postfix"]
        setup_plt()
        df = pd.read_csv("eval_writeup/pre_llm_perf.csv")
        id_vars = ["Dataset", "Threshold"]
        # Filter the dataframe to only contain the selected metrics and id_vars
        df = df[selected_metrics + id_vars]
        # Reshape the dataframe for seaborn
        df_melted = pd.melt(df, id_vars=id_vars, var_name="Metric", value_name="Value")
        df_melted = rename_datasets(df_melted, preserve_sampled=False)
        # Create the relplot
        g = sns.relplot(
            x="Threshold",
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
        p = Path(f"figures/pre_llm/pre_llm_perf_{postfix}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(f"figures/pre_llm_perf_{postfix}.png")

        # Iterate over each group and plot the dataset
        for dataset_name, group_data in df_melted.groupby("Dataset"):
            plot_dataset(group_data, dataset_name, postfix)
