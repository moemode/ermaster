import seaborn as sns
import pandas as pd

from utils import rename_datasets


def plot_dataset(data, dataset_name):
    # Create the relplot for the specified dataset
    g = sns.relplot(
        x="Threshold",
        y="Value",
        hue="Metric",
        kind="line",
        marker="o",
        data=data,
    )

    # Set title for the plot
    g.fig.suptitle(f"{dataset_name}")
    g.savefig(f"figures/{dataset_name}-pre_llm_perf.png")


df = pd.read_csv("eval_writeup/pre_llm_perf.csv")
id_vars = ["Dataset", "Threshold"]
selected_metrics = ["Precision", "Recall", "F1"]

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


# Adjust layout
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Metrics vs Threshold for Different Datasets", fontsize=16)
g.savefig(f"figures/pre_llm_perf.png")

# Iterate over each group and plot the dataset
for dataset_name, group_data in df_melted.groupby("Dataset"):
    plot_dataset(group_data, dataset_name)
