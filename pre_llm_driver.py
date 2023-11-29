from pathlib import Path
import numpy as np
import pandas as pd
from pre_llm import find_matching_csv, pre_llm

CONFIGURATIONS = {
    "base": {"runfiles": "runs/35_base/*force-gpt*.json", "similarities": "eval"},
}

if __name__ == "__main__":
    results = []
    # Create values in the range 0.0 to 1.0 with an increment of 0.05
    threshold_values = np.arange(0.0, 1.05, 0.05)

    # Iterate over the threshold values
    for threshold in threshold_values:
        for path in Path(".").glob(CONFIGURATIONS["base"]["runfiles"]):
            dataset_name = path.stem.split("-")[0]
            simPath = find_matching_csv(
                path, Path(CONFIGURATIONS["base"]["similarities"]).glob("*-allsim.csv")
            )
            if not simPath:
                raise ValueError(
                    f"No matching similarity file in {CONFIGURATIONS['base']['similarities']} found for {path}"
                )

            # Call pre_llm and store the results in the dataframe
            acc, prec, rec, f1, cost, cost_rel, duration, duration_rel = pre_llm(
                threshold, path, simPath
            )
            results.append(
                {
                    "Dataset": dataset_name,
                    "Threshold": threshold,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "Cost": cost,
                    "Cost Relative": cost_rel,
                    "Duration": duration,
                    "Duration Relative": duration_rel,
                }
            )
    results_df = pd.DataFrame(results)
    # Print or further process the results dataframe
    print(results_df)
    results_df.to_csv("eval_writeup/pre_llm_perf.csv", index=False)
