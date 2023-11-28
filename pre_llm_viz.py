import seaborn as sns
import pandas as pd


df = pd.read_csv("eval_writeup/pre_llm_perf.csv")

# Reshape the dataframe for seaborn
df_melted = pd.melt(
    df, id_vars=["Dataset", "Threshold"], var_name="Metric", value_name="Value"
)

# Set seaborn style
sns.set(style="whitegrid")

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
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Metrics vs Threshold for Different Datasets", fontsize=16)
g.savefig(f"figures/pre_llm_perf.png")
