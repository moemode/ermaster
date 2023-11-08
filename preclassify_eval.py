import pandas as pd
from matplotlib import pyplot as plt
from preclassify import count_miss_classifications

if __name__ == "__main__":
    # Specify the path to the CSV file
    file_path = "/home/v/coding/ermaster/eval/structured_itunes_amazon-sim.csv"
    # Read the CSV file into a DataFrame
    s = pd.read_csv(file_path)
    similarity_columns = ["jaccard", "overlap", "mongeelkan", "genjaccard"]
    # Create a dictionary to store the data for each similarity column
    data_dict = {}
    for name in similarity_columns:
        # Calculate miss classifications for each similarity column
        data = count_miss_classifications(s, name)
        data_dict[name] = data

    # Create a graph using Matplotlib with different lines for each dataset
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        x_values, y_values = zip(*data)
        plt.plot(x_values, y_values, label=f"Miss Classifications - {name}")

    plt.xlabel("Row Index")
    plt.ylabel("Count of Miss Classifications")
    plt.title("Miss Classifications vs. Row Index")
    plt.legend()
    plt.grid(True)
    plt.show()
    while True:
        pass
