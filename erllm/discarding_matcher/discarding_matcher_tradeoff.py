"""
Generate and analyze performance trade-off metrics for the discarding matcher based on F1 decrease thresholds.
Reads performance metrics from a CSV file, calculates F1 decrease, relative cost, and relative duration for each dataset and threshold.
"""
import pandas as pd
from erllm import EVAL_FOLDER_PATH, EVAL_WRITEUP_FOLDER_PATH

if __name__ == "__main__":
    df = pd.read_csv(EVAL_FOLDER_PATH / "discarding_matcher_perf.csv")
    decrease_entries = []
    for dataset in df["Dataset"].unique():
        # Filter to current dataset
        dataset_df = df[df["Dataset"] == dataset]
        # Get the entry for threshold 0
        reference_entry = dataset_df[dataset_df["Threshold"] == 0.0].iloc[0]
        # Calculate the decrease in F1 for each threshold and include the relative cost
        dataset_df["F1_Decrease"] = (dataset_df["F1"] - reference_entry["F1"]) / (
            reference_entry["F1"]
        )
        dataset_df["Original F1"] = reference_entry["F1"]
        dataset_df["F1"] = dataset_df["F1"]
        dataset_df["Relative_Cost"] = dataset_df["Cost Relative"]
        dataset_df["Relative_Duration"] = dataset_df["Duration Relative"]
        # Append the results to the list
        decrease_entries.append(
            dataset_df[
                [
                    "Dataset",
                    "Threshold",
                    "F1_Decrease",
                    "Original F1",
                    "F1",
                    "Relative_Cost",
                    "Relative_Duration",
                ]
            ]
        )
    # Concatenate the list of DataFrames into the final result DataFrame
    decrease_df = pd.concat(decrease_entries, ignore_index=True)
    # Print the result DataFrame
    # print(result_df)

    # Values for F1 decrease thresholds
    f1_decrease_thresholds = [-0.05, -0.1, -0.15]
    aboveth_entries = []
    # Iterate through each unique dataset
    for dataset in decrease_df["Dataset"].unique():
        # Filter the DataFrame for the current dataset
        dataset_df = decrease_df[decrease_df["Dataset"] == dataset]
        # Get the rows with the largest threshold where F1 decrease is greater than the threshold
        for threshold in f1_decrease_thresholds:
            largest_threshold_row = (
                dataset_df[(dataset_df["F1_Decrease"] > threshold)]
                .sort_values(by="Threshold", ascending=False)
                .iloc[0]
            )
            aboveth_entries.append(
                largest_threshold_row[
                    [
                        "Dataset",
                        "Threshold",
                        "Original F1",
                        "F1",
                        "F1_Decrease",
                        "Relative_Cost",
                        "Relative_Duration",
                    ]
                ]
                .to_frame()
                .T.assign(F1_Decrease_Threshold=threshold)
            )
    # Concatenate the list of DataFrames into the final result DataFrame
    above_th = pd.concat(aboveth_entries, ignore_index=True)
    presentation_df = above_th.copy()
    # add new column 1 - relative cost
    presentation_df["Cost Decrease"] = 1 - presentation_df["Relative_Cost"]
    presentation_df["F1_Decrease"] = presentation_df["F1_Decrease"].abs()
    presentation_df["Time Decrease"] = 1 - presentation_df["Relative_Duration"]

    print(
        presentation_df[
            [
                "Dataset",
                "F1_Decrease",
                "Cost Decrease",
                "Time Decrease",
                "F1_Decrease_Threshold",
            ]
        ]
    )
    presentation_df.to_csv(
        EVAL_WRITEUP_FOLDER_PATH / "discarding_matcher_tradeoff.csv", index=False
    )
