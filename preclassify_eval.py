import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns


def count_miss_classifications(similarities: pd.DataFrame, name: str):
    # Sort the DataFrame by the specified column in ascending order
    similarities.sort_values(by=name, ascending=True, inplace=True)
    match_count = 0
    miss_classifiation_function = []
    for i, (_, row) in enumerate(similarities.iterrows()):
        if row["label"] == 1:
            match_count += 1
        miss_classifiation_function.append((i + 1, match_count))
    return miss_classifiation_function


if __name__ == "__main__":
    # Specify the path to the CSV file
    file_paths = Path("/home/v/coding/ermaster/eval").glob("*-sim.csv")
    datasets = dict()
    # Define a list to store the data
    data_list = []
    for f in file_paths:
        # Read the CSV file into a DataFrame
        ds = f.stem.split("-")[0]
        s = pd.read_csv(f)
        similarity_columns = ["jaccard", "overlap", "mongeelkan", "genjaccard"]
        # Create a dictionary to store the data for each similarity column
        data_dict = {}
        for name in similarity_columns:
            # Calculate miss classifications for each similarity column
            data = count_miss_classifications(s, name)
            for x, y in data:
                data_list.append([ds, name, x, y])
            data_dict[name] = data
        datasets[ds] = data_dict
        # Create a graph using Matplotlib with different lines for each dataset
        plt.figure(figsize=(10, 6))
        for name, data in data_dict.items():
            x_values, y_values = zip(*data)
            plt.plot(x_values, y_values, label=f"{name}")
        plt.xlabel("# Discarded")
        plt.ylabel("# False Negatives")
        plt.title(f"Discarding in order of similarity on {ds}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/{ds}-miss-classifications.png")
    df = pd.DataFrame(
        data_list, columns=["dataset", "measure", "n_discarded", "n_false_negatives"]
    )
    df.to_csv("eval/missclassifications.csv", index=False)
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
    g.savefig("figures/missclassifications.png")
    print(df)
