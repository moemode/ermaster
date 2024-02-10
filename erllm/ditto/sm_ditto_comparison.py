import pandas as pd
from erllm import EVAL_FOLDER_PATH


if __name__ == "__main__":
    sm_results = pd.read_csv(
        EVAL_FOLDER_PATH / "discarding_selective_matcher/grid/result.csv"
    )
    # keep entries where Label Fraction is 0.0 and Discard Fraction is 0.15
    sm_results = sm_results[
        (sm_results["Label Fraction"] == 0.0) & (sm_results["Discard Fraction"] == 0.15)
    ]
    ditto_results = pd.read_csv(
        EVAL_FOLDER_PATH / "ditto/classification_performance.csv"
    )
    sm_results["Method"] = "SM"
    ditto_results["Method"] = "Ditto"
    sm_results = sm_results[["Dataset", "Method", "F1", "Precision", "Recall"]]
    ditto_results = ditto_results[["Dataset", "Method", "F1", "Precision", "Recall"]]
    # remove row where Dataset is "dbpedia"
    ditto_results = ditto_results[ditto_results["Dataset"] != "dbpedia"]
    results = pd.concat([sm_results, ditto_results])
    # for each dataset there is a row for SM and Ditto
    pivot_table = results.pivot_table(
        index="Dataset", columns=["Method"], values=["F1", "Precision", "Recall"]
    )
    print(pivot_table)
