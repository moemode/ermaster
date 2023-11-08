from pathlib import Path
import pandas as pd
from typing import List, Tuple

from access_dbpedia import OrderedEntity


def load_into_df(fpaths: List[Path]) -> pd.DataFrame:
    """
    Load multiple CSV files into a single pandas DataFrame and ensure they have the same shape.
    Args:
        fpaths (List[Path]): A list of file paths to the CSV files to be loaded.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the concatenated data from all CSV files.
    Raises:
        ValueError: If no file paths are provided, or if a CSV file has a different structure compared to the first file.

    The function loads the first CSV file in the list to determine the initial structure of the DataFrame.
    It then iterates through the remaining file paths, loading each CSV file, and checks if their column structure matches
    the structure of the initial DataFrame. If a column structure mismatch is detected, a ValueError is raised.
    All valid CSV files are concatenated into a single DataFrame. The '_id' column is assumed to be present in each file
    and is set as the index of the resulting DataFrame.

    Example usage:
    file_paths = [Path("file1.csv"), Path("file2.csv"), Path("file3.csv")]
    resulting_dataframe = load_into_df(file_paths)
    """
    if not fpaths:
        raise ValueError("No file paths provided.")
    first_file_path = fpaths[0]
    df = pd.read_csv(first_file_path)
    # Loop through the remaining file paths and validate their shape
    for file_path in fpaths[1:]:
        # Load the CSV file
        new_df = pd.read_csv(file_path)
        # Check if the columns match the structure of the initial DataFrame
        if not df.columns.equals(new_df.columns):
            raise ValueError(f"CSV file at {file_path} has a different structure.")
        # Concatenate the new DataFrame to the existing DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
    df.set_index("_id", inplace=True)
    return df


def load_benchmark(
    fpaths: List[Path],
) -> List[Tuple[bool, OrderedEntity, OrderedEntity]]:
    """
    Load benchmark data from CSV files and return a list of tuples representing entity pairs.

    Args:
        fpaths (List[Path]): A list of file paths to the CSV files containing benchmark data.

    Returns:
        List[Tuple[bool, OrderedEntity, OrderedEntity]]: A list of tuples, each containing three elements:
            1. A boolean value representing the label for the entity pair.
            2. An OrderedEntity object representing the first entity.
            3. An OrderedEntity object representing the second entity.

    The function loads benchmark data from CSV files specified by the file paths in the 'fpaths' list. It assumes that
    the CSV files follow a specific structure where two tables, 'table1' and 'table2,' are compared, and the labels
    for the entity pairs are included.

    The 'OrderedEntity' objects are created for each entity in both tables, and the entity pairs are represented as tuples.
    The 'OrderedEntity' objects include the entity's ID and attributes extracted from the corresponding columns in the CSV.

    Example usage:
    file_paths = [Path("benchmark1.csv"), Path("benchmark2.csv")]
    benchmark_data = load_benchmark(file_paths)
    for label, entity1, entity2 in benchmark_data:
        # Process entity pairs and their labels
        ...
    """
    # Read the CSV file into a Pandas DataFrame
    df = load_into_df(fpaths)
    # Display the first few rows of the DataFrame
    print(df.head())
    print(df.columns)
    # Iterate over the rows of the DataFrame
    table1_columns = [col for col in df.columns if col.startswith("table1")]
    column_names = [col.split(".")[1] for col in table1_columns]
    table2_columns = [col for col in df.columns if col.startswith("table2")]
    assert list(map(lambda n: "table2." + n, column_names)) == table2_columns
    columns_no_id = [name for name in column_names if name != "id"]
    table1_columns_no_id = [col for col in table1_columns if col != "table1.id"]
    table2_columns_no_id = [col for col in table2_columns if col != "table2.id"]
    table1_entities = []
    table2_entities = []
    pairs: List[Tuple[bool, OrderedEntity, OrderedEntity]] = []
    for _, row in df.iterrows():
        attributes1 = zip(columns_no_id, map(str, row[table1_columns_no_id].values))
        attributes2 = zip(columns_no_id, map(str, row[table2_columns_no_id].values))
        e1 = OrderedEntity(row["table1.id"], None, attributes1)
        e2 = OrderedEntity(row["table2.id"], None, attributes2)
        table1_entities.append(e1)
        table2_entities.append(e2)
        pairs.append((row["label"], e1, e2))
    return pairs


if __name__ == "__main__":
    fnames = ["test.csv", "train.csv", "valid.csv"]
    folder = Path(
        "/home/v/coding/ermaster/data/benchmark_datasets/existingDatasets/structured_itunes_amazon"
    )
    pairs = load_benchmark([folder / fname for fname in fnames])
    print(pairs[0])
    print(len(pairs))
