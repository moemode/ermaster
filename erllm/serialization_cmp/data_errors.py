import pandas as pd
from erllm import EVAL_FOLDER_PATH
from eval_writeup.writeup_utils import rename_datasets


if __name__ == "__main__":
    matcher_files = EVAL_FOLDER_PATH / "llm_matcher"
    embed_1 = pd.read_csv(matcher_files / "base_wattr_names_embed_one_ppair.csv")
    embed_half = pd.read_csv(matcher_files / "base_wattr_names_embed_half.csv")
    misfield_half = pd.read_csv(matcher_files / "base_wattr_names_misfield_half.csv")
    misfield_all = pd.read_csv(matcher_files / "base_wattr_names_misfield_all.csv")
    attr = pd.read_csv(matcher_files / "base_wattr_names.csv")
    dfs = [embed_1, embed_half, misfield_half, misfield_all, attr]
    df_scheme = list(
        zip(dfs, ["embed-1", "embed-50%", "misfield-50%", "misfield-100%", "attr"])
    )
    data = []
    for df, scheme in df_scheme:
        df["Scheme"] = scheme
        data.append(df[["Dataset", "Precision", "Recall", "F1", "Scheme"]])
    df = pd.concat(data)
    # throw out any results where Dataset contains 'dpedia'
    df = df[~df["Dataset"].str.lower().str.contains("dbpedia")]
    df = rename_datasets(df, preserve_sampled=False)
    # Reshape the DataFrame

    mean_table = df.groupby("Scheme")[["Precision", "Recall", "F1"]].mean().transpose()
    mean_table = mean_table[
        ["attr", "embed-1", "embed-50%", "misfield-50%", "misfield-100%"]
    ]
    mean_table = mean_table.reindex(["F1", "Precision", "Recall"])

    mean_table.columns.name = None
    s = mean_table.style
    s.format(precision=2)
    s.format_index(axis=1, escape="latex")
    """
    s.highlight_max(
        props="font-weight: bold",
        axis=1,
    )
    """
    latex_table = s.to_latex(
        EVAL_FOLDER_PATH / "serialization_cmp" / f"embed_misfield_means.tex",
        column_format="lccccc",
        hrules=True,
        convert_css=True,
        position_float="centering",
        caption="Mean F1, Precision and Recall across datasets for LLM Matcher (gpt-3.5-turbo-instruct) using base prompt prefix with attribute names and data errors.",
        label="tab:de-mean-cmp",
    )
    print(mean_table)
    mean_diff_table = mean_table.copy()
    diff_columns = mean_table.columns[1:]
    for col in diff_columns:
        new_col_name = f"{col}_diff"
        mean_diff_table[new_col_name] = mean_table[col] - mean_table["attr"]
    print(mean_diff_table)
