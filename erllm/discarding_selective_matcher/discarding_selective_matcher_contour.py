from pathlib import Path
import pandas as pd
from erllm import EVAL_FOLDER_PATH
import matplotlib.pyplot as plt
import numpy as np

CONFIGURATIONS = {
    "basic-cmp": {
        "result_folder": EVAL_FOLDER_PATH
        / "discarding_selective_matcher"
        / "basic_cmp",
        "label_fractions": [0, 0.05, 0.1, 0.15],
        "mean_metrics": ["F1", "Precision", "Recall", "Accuracy"],
    },
    "grid": {
        "result_folder": EVAL_FOLDER_PATH / "discarding_selective_matcher" / "grid",
        "mean_metrics": ["F1", "Precision", "Recall", "Accuracy"],
        "max_label_fraction": 0.15,
    },
}


def format_percentages(c):
    return f"{c*100:.0f}\%"


def get_mean_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.groupby(["Label Fraction", "Discard Fraction"])
    # Calculate mean for each metric
    df = df[metric].mean().reset_index()
    return df.pivot(index="Discard Fraction", columns="Label Fraction", values=metric)


def make_contour_2d(df: pd.DataFrame, metric: str, save_to: Path) -> str:
    pivot_df = get_mean_df(df, metric)
    x = pivot_df.columns.to_numpy()
    y = pivot_df.index.to_numpy()
    X, Y = np.meshgrid(x, y)
    plt.figure()
    # Increase color resolution by specifying more levels
    n_levels = 20  # You can adjust this value based on your preference
    plt.figure()
    levels = np.linspace(0, 1, n_levels + 1)
    contours = plt.contourf(X, Y, pivot_df.values, levels=levels, cmap="viridis")
    plt.xlabel("Label Fraction")
    plt.ylabel("Discard Fraction")
    plt.colorbar(contours, label=metric)
    plt.title(f"Contour Plot for {metric}")
    file_path = save_to / f"contour_plot_{metric}.png"
    plt.savefig(file_path)


def make_contour_2d_im(df: pd.DataFrame, metric: str, save_to: Path) -> str:
    pivot_df = get_mean_df(df, metric)
    x = pivot_df.columns.to_numpy()
    y = pivot_df.index.to_numpy()
    X, Y = np.meshgrid(x, y)
    plt.figure()
    im = plt.imshow(pivot_df.values, cmap="viridis", extent=[x[0], x[-1], y[-1], y[0]])
    plt.xlabel("Label Fraction")
    plt.ylabel("Discard Fraction")
    plt.colorbar(im, label=metric)
    plt.title(f"Heatmap for {metric}")
    plt.show()


def make_contour_3d(df: pd.DataFrame, metric: str, save_to: Path) -> str:
    pivot_df = get_mean_df(df, metric)
    print(pivot_df)
    x = pivot_df.columns.to_numpy()
    y = pivot_df.index.to_numpy()
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, pivot_df.values, cmap="viridis")
    # Customize the plot
    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("Discard Fraction")
    ax.set_zlabel(metric)
    ax.set_title(f"Meshgrid for {metric}")
    plt.show()


if __name__ == "__main__":
    cfg_name = "grid"
    cfg = CONFIGURATIONS[cfg_name]
    df = pd.read_csv(cfg["result_folder"] / "result.csv")
    # filter up to max_label_fraction
    if "max_label_fraction" in cfg:
        df = df[df["Label Fraction"] <= cfg["max_label_fraction"]]
    # check if cfg has label_fractions
    if "label_fractions" in cfg:
        df = df[df["Label Fraction"].isin(cfg["label_fractions"])]
    for m in cfg["mean_metrics"]:
        make_contour_2d(df, m, cfg["result_folder"])
