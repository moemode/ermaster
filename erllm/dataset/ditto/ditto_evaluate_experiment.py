"""
Based on the stats of train and valid set and the results of running ditto,
calculate the precision, recall, and F1 score for the total dataset.
"""

from pathlib import Path
import pandas as pd
from erllm import DATA_FOLDER_PATH, EVAL_FOLDER_PATH
from erllm.dataset.ditto.ditto_stats import ditto_stats
from erllm.utils import f1, precision, recall


if __name__ == "__main__":
    testset_results = pd.read_csv(EVAL_FOLDER_PATH / "ditto" / "runs.csv")
    classification_performance = []
    # iterate over each row representing a result on the test set of a dataset
    for _, row in testset_results.iterrows():
        ds_name = row["Runtag"]
        ds_name = ds_name.split("_lm")[0]
        ditto_ds_folder = Path(DATA_FOLDER_PATH / f"benchmark_datasets/ditto/{ds_name}")
        n_train, n_train_tp, n_traing_tn = ditto_stats(ditto_ds_folder / "train.txt")
        n_valid, n_valid_tp, n_valid_tn = ditto_stats(ditto_ds_folder / "valid.txt")
        n_test_tp, n_test_tn, n_test_fp, n_test_fn = (
            row["Test TP"],
            row["Test TN"],
            row["Test FP"],
            row["Test FN"],
        )
        n_test = n_test_tp + n_test_tn + n_test_fp + n_test_fn
        # assumes that valid and test set contain purely correct labels
        n, tp, tn, fp, fn = (
            n_train + n_valid + n_test,
            n_train_tp + n_valid_tp + n_test_tp,
            n_traing_tn + n_valid_tn + n_test_tn,
            n_test_fp,
            n_test_fn,
        )
        classification_performance.append(
            {
                "Dataset": ds_name,
                "N": n,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": precision(tp, fp),
                "Recall": recall(tp, fn),
                "F1": f1(tp, tn, fp, fn),
            }
        )
    perf_df = pd.DataFrame(classification_performance)
    perf_df.to_csv(
        EVAL_FOLDER_PATH / "ditto" / "classification_performance.csv", index=False
    )
