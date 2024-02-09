import pandas as pd
from erllm import EVAL_FOLDER_PATH


sm_results = pd.read_csv(
    EVAL_FOLDER_PATH / "discarding_selective_matcher/grid/result.csv"
)
# keep entries where Label Fraction is 0.0 and Discard Fraction is 0.15
sm_results = sm_results[
    (sm_results["Label Fraction"] == 0.0) & (sm_results["Discard Fraction"] == 0.15)
]
print(sm_results)

ditto_results = pd.read_csv(EVAL_FOLDER_PATH / "ditto/classification_performance.csv")
print(ditto_results)
