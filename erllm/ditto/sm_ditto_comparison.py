"""
Creates a table containing F1 scores for DITTO and SM across all datasets.
"""

import pandas as pd
from erllm import EVAL_FOLDER_PATH
from erllm.utils import rename_datasets


cfg = {
    "f1": {"columns": [f"F1"]},
    "all_metrics": {"columns": ["Precision", "Recall", "F1"]},
}

if __name__ == "__main__":
    for cfg_name, cfg in cfg.items():
        sm_results = pd.read_csv(
            EVAL_FOLDER_PATH / "discarding_selective_matcher/grid/result.csv"
        )
        # keep entries where Label Fraction is 0.0 and Discard Fraction is 0.15
        sm_results = sm_results[
            (sm_results["Label Fraction"] == 0.15)
            & (sm_results["Discard Fraction"] == 0.0)
        ]
        ditto_results = pd.read_csv(
            EVAL_FOLDER_PATH / "ditto/classification_performance.csv"
        )
        sm_results["Method"] = "SM"
        ditto_results["Method"] = "Ditto"
        sm_results = sm_results[["Dataset", "Method", "F1", "Precision", "Recall"]]
        ditto_results = ditto_results[
            ["Dataset", "Method", "F1", "Precision", "Recall"]
        ]
        # remove row where Dataset is "dbpedia"
        ditto_results = ditto_results[ditto_results["Dataset"] != "dbpedia"]
        results = pd.concat([sm_results, ditto_results])
        results = rename_datasets(results, False)
        # for each dataset there is a row for SM and Ditto
        pivot_table = results.pivot_table(
            index="Dataset", columns=["Method"], values=cfg["columns"]
        )
        pivot_table = pivot_table.reindex(["SM", "Ditto"], axis=1, level=1)
        # order decreasingly in terms of F1 of SM
        pivot_table = pivot_table.sort_values(by=("F1", "SM"), ascending=False)
        # Add a row  "All" with the mean of the F1, Precision and Recall
        pivot_table.loc["All"] = pivot_table.mean()
        s = pivot_table.style
        s.format(precision=2)
        latex_table = s.to_latex(
            EVAL_FOLDER_PATH / "ditto" / f"{cfg_name}_comparison_table.tex",
            hrules=True,
            position_float="centering",
            multicol_align="c",
            caption=f"Comparison of classification performance",
        )
