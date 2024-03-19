from typing import List
import pandas as pd
from erllm import EVAL_FOLDER_PATH


CONFIGURATIONS = {
    "dbpedia": {
        "full_folder": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "dbpedia_full",
        "sampled_folder": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "grid",
        "dsm_configs": [(0.0, 0.0), (0.8, 0.0), (0.0, 0.15), (0.5, 0.15)],
    },
}


def combine_results(
    full_df: pd.DataFrame, sample_df: pd.DataFrame, dsm_configs: List[tuple[int, int]]
) -> pd.DataFrame:
    # get all datasets in full_df
    full_datasets = list(full_df["Dataset"].unique())
    sampled_datasets = list(sample_df["Dataset"].unique())
    keep_sampled_datasets = filter(
        lambda ds: any([full_ds in ds for full_ds in full_datasets]), sampled_datasets
    )
    # maps each dataset (also full ones) to the full one
    ds_to_full = {
        ds: fds
        for fds in full_datasets
        for ds in sampled_datasets + full_datasets
        if fds in ds
    }
    # only keep rows in sample_df where the Dataset column is contained in any of the strings in full_datasets
    sample_df = sample_df[sample_df["Dataset"].isin(keep_sampled_datasets)]
    # only keep rows where Discard Fraction, Label Fraction is in cfg["dsm_configs"]
    full_df = full_df[
        full_df[["Discard Fraction", "Label Fraction"]].apply(
            lambda row: tuple(row) in dsm_configs, axis=1
        )
    ]
    sample_df = sample_df[
        sample_df[["Discard Fraction", "Label Fraction"]].apply(
            lambda row: tuple(row) in dsm_configs, axis=1
        )
    ]
    # full_result keep Dataset , Discard Fraction, Label Fraction, Recall, Precision, F1
    full_result = full_df[
        ["Dataset", "Discard Fraction", "Label Fraction", "Recall", "Precision", "F1"]
    ]
    full_result["Is Full"] = True
    sample_result = sample_df[
        ["Dataset", "Discard Fraction", "Label Fraction", "Recall", "Precision", "F1"]
    ]
    sample_result["Is Full"] = False
    # combine the two df into one
    result = pd.concat([full_result, sample_result])
    # add column "Full Dataset"
    result["Full Dataset"] = result["Dataset"].map(ds_to_full)
    # result = result.set_index(["Dataset", "Discard Fraction", "Label Fraction"])
    return result


if __name__ == "__main__":
    cfg = CONFIGURATIONS["dbpedia"]
    full_df = pd.read_csv(cfg["full_folder"] / "result.csv")
    sample_df = pd.read_csv(cfg["sampled_folder"] / "result.csv")
    r = combine_results(full_df, sample_df, cfg["dsm_configs"])
    print(r)
