import pandas as pd

df = pd.read_csv("eval_writeup/discarding_matcher_duration_cmp.csv")
df.to_latex(
    "eval_writeup/discarding_matcher_duration_cmp.ltx", escape=True, index=False
)
