import pandas as pd
from pathlib import Path
import seaborn as sns


def count_miss_classifications(similarities: pd.DataFrame, sim_name: str):
    # Sort the DataFrame by the specified column in ascending order
    similarities.sort_values(by=sim_name, ascending=True, inplace=True)
    N_total = len(similarities)
    n_fn = 0
    stats = [(0, 0, 0, 0, 0)]
    for i, (_, row) in enumerate(similarities.iterrows()):
        if row["label"] == 1:
            n_fn += 1
        coverage = (i + 1) / N_total
        risk = n_fn / (i + 1)
        fpr = n_fn / N_total
        stats.append((i + 1, n_fn, coverage, risk, fpr))
    return stats


if __name__ == "__main__":
    # Specify the path to the CSV file
    file_paths = Path("/home/v/coding/ermaster/eval").glob("*-allsim.csv")
    datasets = dict()
    data_list = []

    for f in file_paths:
        # Read the CSV file into a DataFrame
        ds = f.stem.split("-")[0]
        s = pd.read_csv(f)
        similarity_columns = [
            "jaccard",
            "overlap",
            "mongeelkan",
            "genjaccard",
            "cosine_sim",
            "euclidean_sim",
        ]
        # Create a dictionary to store the data for each similarity column
        data_dict = {}
        for sim_name in similarity_columns:
            # Calculate miss classifications for each similarity column
            # check if name exists in dataframe
            if sim_name not in s.columns:
                continue
            stats = count_miss_classifications(s, sim_name)
            for n, n_fn, coverage, risk, fpr in stats:
                data_list.append([ds, sim_name, n, n_fn, coverage, risk, fpr])
            data_dict[sim_name] = data_list
        """
        # Create a graph using Matplotlib with different lines for each dataset
        fig = plt.figure(figsize=(10, 6))
        for sim_name, n_fn in data_dict.items():
            x_values, y_values = zip(*n_fn)
            plt.plot(x_values, y_values, label=f"{sim_name}")
            plt.xlabel("# Discarded")
            plt.ylabel("# False Negatives")
            plt.title(f"Discarding in order of similarity on {ds}")
            plt.legend()
            plt.grid(True)
            plt.close(fig)
            plt.savefig(f"figures/{ds}-miss-classifications.png")
        datasets[ds] = data_dict
        """
    df = pd.DataFrame(
        data_list,
        columns=[
            "dataset",
            "measure",
            "n_discarded",
            "n_false_negatives",
            "coverage",
            "risk",
            "fpr",
        ],
    )
    df.to_csv("eval/cr-missclassifications.csv", index=False)
    g = sns.relplot(
        data=df,
        x="n_discarded",
        y="n_false_negatives",
        hue="measure",
        col="dataset",
        kind="line",
        col_wrap=3,
        facet_kws={"sharex": False, "sharey": False},
    )
    # Set axis labels
    g.set_axis_labels("# Discarded", "# False Negatives")
    # Save the Seaborn plot to a file
    g.savefig("figures/cr-missclassifications.png")
    print(df)
