from pathlib import Path

DATA_FOLDER_PATH = Path(__file__).resolve().parent.parent / "data"
DBFILE_PATH = DATA_FOLDER_PATH / "my_database.db"
DATASET_FOLDER_PATH = DATA_FOLDER_PATH / Path("benchmark_datasets/existingDatasets")
PROMPT_DATA_FOLDER_PATH = DATA_FOLDER_PATH / Path("prompt_data")
PROMPTS_FOLDER_PATH = DATA_FOLDER_PATH / Path("prompts")
RUNS_FOLDER_PATH = DATA_FOLDER_PATH / Path("runs")
EVAL_WRITEUP_FOLDER_PATH = Path(__file__).resolve().parent.parent / "eval_writeup"

ORIGINAL_DATASET_NAMES = [
    # "dirty_dblp_acm",
    # "dirty_walmart_amazon",
    # "dirty_itunes_amazon",
    # "dirty_dblp_scholar",
    "structured_dblp_acm",
    "structured_itunes_amazon",
    # "textual_company",
    "structured_amazon_google",
    "structured_dblp_scholar",
    "structured_walmart_amazon",
    "structured_beer",
    "structured_fodors_zagats",
    "textual_abt_buy",
    "dbpedia10k",
    "dbpedia10k_harder",
]
SAMPLED_DATASET_NAMES = [dataset + "_1250" for dataset in ORIGINAL_DATASET_NAMES]

DATASET_NAMES = ORIGINAL_DATASET_NAMES + SAMPLED_DATASET_NAMES
