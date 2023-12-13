import numpy as np
from tabulate import tabulate

logprobs = {
    "\n": -0.19837844,
    "Yes": -1.8871049,
    "No": -4.0493307,
    " Yes": -5.0334177,
    " \n": -6.1253004,
}

probs = {k: np.exp(v) for k, v in logprobs.items()}
# probs["total"] = sum(probs.values())
# print as probabilities with 2 decimal places
# print({k: round(100 * v, 2) for k, v in probs.items()})

probs_percent = {k: round(100 * v, 2) for k, v in probs.items()}
table = tabulate(
    [("\qq{k}") for k, v in probs_percent.items()],
    headers=["Token", "Probability (%)"],
    tablefmt="latex_raw",
)

print(table)
