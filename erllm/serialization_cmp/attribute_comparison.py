import pandas as pd
from erllm import EVAL_FOLDER_PATH
from erllm.utils import rename_datasets


if __name__ == "__main__":
    matcher_files = EVAL_FOLDER_PATH / "llm_matcher"
    no_attr = pd.read_csv(matcher_files / "base.csv")
    attr = pd.read_csv(matcher_files / "base_wattr_names.csv")
    attr_rnd = pd.read_csv(matcher_files / "base_wattr_names_rnd_order.csv")
    # only keep Dataset, Precision, Recall, F1
    no_attr = no_attr[["Dataset", "Precision", "Recall", "F1"]]
    no_attr["Scheme"] = "no-attr"
    attr = attr[["Dataset", "Precision", "Recall", "F1"]]
    attr["Scheme"] = "attr"
    attr_rnd = attr_rnd[["Dataset", "Precision", "Recall", "F1"]]
    attr_rnd["Scheme"] = "attr-rnd"
    # create one dataframe with all the results
    df = pd.concat([no_attr, attr, attr_rnd])
    # throw out any results where Dataset contains 'dpedia'
    df = df[~df["Dataset"].str.lower().str.contains("dbpedia")]
    df = rename_datasets(df, preserve_sampled=False)
    # Reshape the DataFrame
    all = df.pivot_table(
        values=["Recall", "Precision", "F1"], index="Dataset", columns="Scheme"
    )
    all.columns.names = [None, None]
    all = all.reindex(["no-attr", "attr", "attr-rnd"], axis=1, level=1)
    all = all.sort_values(by=("F1", "no-attr"), ascending=False)
    all = all.reset_index()
    # make index ordinary column
    s = all.style
    # hide index
    s.hide(axis="index")
    s.format(precision=2)
    (EVAL_FOLDER_PATH / "serialization_cmp").mkdir(parents=True, exist_ok=True)
    latex_table = s.to_latex(
        EVAL_FOLDER_PATH / "serialization_cmp" / f"comparison_table.tex",
        column_format="|l|ccc|ccc|ccc|",
        hrules=True,
        position_float="centering",
        multicol_align="c|",
        caption=f"Comparison of classification performance",
    )
    f1 = df.pivot_table(values=["F1"], index="Dataset", columns="Scheme")
    f1 = f1.reindex(["no-attr", "attr", "attr-rnd"], axis=1, level=1)
    f1 = f1.sort_values(by=("F1", "no-attr"), ascending=False)
    f1 = f1.reset_index()
    s = f1.style
    s.hide(axis="index")
    s.format(precision=2)
    s.highlight_max(
        props="font-weight: bold",
        axis=1,
        subset=["F1"],
    )
    print(f1)
    latex_table = s.to_latex(
        EVAL_FOLDER_PATH / "serialization_cmp" / f"f1_comparison_table.tex",
        column_format="lccc",
        hrules=True,
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        caption=f"F1 scores for LLM Matcher (gpt-3.5-turbo-instruct) using base prompt prefix with various serialization schemes (no-attr, attr, and attr-rnd).",
        label="tab:attr-f1-cmp",
    )
    mean_table = df.groupby("Scheme")[["Precision", "Recall", "F1"]].mean().transpose()
    mean_table = mean_table[["no-attr", "attr", "attr-rnd"]]
    mean_table = mean_table.reindex(["F1", "Precision", "Recall"])
    mean_table.columns.name = None
    s = mean_table.style
    s.format(precision=2)
    s.highlight_max(
        props="font-weight: bold",
        axis=1,
    )
    latex_table = s.to_latex(
        EVAL_FOLDER_PATH / "serialization_cmp" / f"mean_comparison_table.tex",
        column_format="lccc",
        hrules=True,
        convert_css=True,
        position_float="centering",
        caption="Mean Precision, Recall and F1 scores across datasets (except DBpedia) for LLM Matcher (gpt-3.5-turbo-instruct) using base prompt prefix with various serialization schemes (no-attr, attr, and attr-rnd).",
        label="tab:attr-mean-cmp",
    )
    print(mean_table)

    """

    # Melt the DataFrame
    melted_df = pd.melt(
        df, id_vars=["Dataset", "Scheme"], value_vars=["F1", "Recall", "Precision"]
    )

    # Create pivot table with multi-index columns
    pivot_table = melted_df.pivot_table(
        values="value", index=["Dataset", "variable"], columns="Scheme"
    )
    melted_df = pd.melt(
        df, id_vars=["Dataset", "Scheme"], value_vars=["F1", "Recall", "Precision"]
    )
    # for each datasetcreate pivot table for F1
    pivot_table = pd.pivot_table(
        df, values=["F1", "Recall", "Precision"], index=["Dataset", "Scheme"]
    )
    print(pivot_table)
    # create table of mean Precision, Recall, F1 for each scheme
    mean_table = df.groupby("Scheme")[["Precision", "Recall", "F1"]].mean()
    print(mean_table)
    """
