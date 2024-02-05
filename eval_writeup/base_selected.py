import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
from writeup_utils import rename_datasets

configurations = {
    "base": {
        "fname": "base.csv",
    },
    "base-wattr-names": {
        "fname": "base_wattr_names.csv",
    },
    "base-wattr-names-rnd-order": {
        "fname": "base_wattr_names_rnd_order.csv",
    },
    "base-wattr-names-embed-05": {
        "fname": "base_wattr_names_embed_05.csv",
    },
}

cfg = configurations["base-wattr-names-embed-05"]
# Read in the CSV file into a DataFrame
df = pd.read_csv(f"eval/llm_matcher/{cfg['fname']}")

df = df[
    [
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        # "Accuracy",
    ]
]
outfname = Path(cfg["fname"]).stem + "_selected.csv"

df = rename_datasets(df, preserve_sampled=False)
df.to_csv(f"eval_writeup/{outfname}", index=False)
