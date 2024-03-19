from typing import List
import pandas as pd
from erllm import EVAL_FOLDER_PATH
from erllm.utils import rename_datasets


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
    full_result["Type"] = "Full"
    sample_result = sample_df[
        ["Dataset", "Discard Fraction", "Label Fraction", "Recall", "Precision", "F1"]
    ]
    sample_result["Type"] = "Sampled"
    # combine the two df into one
    result = pd.concat([full_result, sample_result])
    # add column "Full Dataset"
    result["Full Dataset"] = result["Dataset"].map(ds_to_full)
    # result = result.set_index(["Dataset", "Discard Fraction", "Label Fraction"])
    return result


def comparison_table(combined_df: pd.DataFrame, dsm_configs: List[tuple[int, int]]):
    p = combined_df.pivot_table(
        index=["Full Dataset", "Discard Fraction", "Label Fraction"],
        columns="Type",
        values=["Recall", "Precision", "F1"],
    )
    p.reset_index(inplace=True)
    # reindex so that "Recall", "Precision", "F1" are in that order
    p = p.reindex(
        [
            "Full Dataset",
            "Discard Fraction",
            "Label Fraction",
            "Recall",
            "Precision",
            "F1",
        ],
        axis=1,
        level=0,
    )
    p = p.reindex(["Sampled", "Full", ""], axis=1, level=1)
    p.rename(
        columns={
            "Full Dataset": "Dataset",
            "Discard Fraction": "Discard",
            "Label Fraction": "Label",
        },
        inplace=True,
    )
    p["Cfg Number"] = p[["Discard", "Label"]].apply(
        lambda row: dsm_configs.index(tuple(row)), axis=1
    )
    p = p.sort_values(by=["Dataset", "Cfg Number"])
    p.insert(0, "Configuration", "Example")
    # add column Configuration to very left
    # apply rename_dataset function to column Dataset
    p = rename_datasets(p)
    s = p.style
    s.format(
        lambda s: "{:.0f}\%".format(s * 100),
        subset=p.columns.get_loc_level("Discard", level=0)[0],
    )
    s.format(
        lambda s: "{:.0f}\%".format(s * 100),
        subset=p.columns.get_loc_level("Label", level=0)[0],
    )
    s.format(precision=2, subset=["F1", "Precision", "Recall"])
    s.hide()
    s.hide(axis=1, subset="Cfg Number")
    latex_table = s.to_latex(
        cfg["full_folder"] / "full_sampled_table.tex",
        # column_format="lccc",
        hrules=True,
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        caption="Classification performance of various DSM configurations on the DBpedia dataset",
        label="tab:dbpedia-full-sampled",
    )


if __name__ == "__main__":
    cfg = CONFIGURATIONS["dbpedia"]
    full_df = pd.read_csv(cfg["full_folder"] / "result.csv")
    sample_df = pd.read_csv(cfg["sampled_folder"] / "result.csv")
    r = combine_results(full_df, sample_df, cfg["dsm_configs"])
    comparison_table(r, cfg["dsm_configs"])
