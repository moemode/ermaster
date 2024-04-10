def rename_datasets(df, preserve_sampled=True):
    # Remove prefixes from Dataset column
    df["Dataset"] = df["Dataset"].str.replace(r"^structured_|textual_", "", regex=True)
    # Reorder columns for clarity
    df["Dataset"] = df["Dataset"].str.replace("_", "-").str.title()
    df["Dataset"] = df["Dataset"].str.replace(
        "-1250", " Sampled" if preserve_sampled else ""
    )
    df["Dataset"] = df["Dataset"].str.replace("Dbpedia10K", "DBpedia")
    return df
