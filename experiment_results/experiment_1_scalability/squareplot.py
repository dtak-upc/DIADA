import math
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt

ALGORITHMS = ("ours", "Apriori", "FP-Growth")
DATASETS = ("Jannis", "Miniboone", "Drive Diagnosis")

# dataset_name -> algo -> {'row': {'x': [...], 'y': [...]}, 'col': {'x': [...], 'y': [...]}}
SquareData = Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]

RESULTS_PATH = "results.pickle"
SQUAREPLOT_PATH = "squareplot.png"

ALGO_NAME_OVERRIDES = {"LIMA": "DIADA"}
DATASET_NAME_OVERRIDES = {"drive": "drive diagnosis"}

FONT_SIZE = 18

# (xlim, ylim) per row: row-scalability (aproxInv), then column-scalability
ROW_LIMITS = ((1e3, 1e8), (0, 300))
COL_LIMITS = ((10, 100), (0, 300))

ROW_X_LABEL = r"$\epsilon^{-1}$"
COL_X_LABEL = "columns"
Y_LABEL = "time (s)"
ROW_SIDE_LABELS = ("row scalability", "column scalability")


def plotSquare(data: SquareData, save_path: str = SQUAREPLOT_PATH) -> None:
    """A 2 x len(data) grid: one column per dataset, row 0 = row-scalability,
    row 1 = column-scalability, each subplot with one line per algorithm.

    Each row of subplots is contiguous and shares a single left y-axis
    (only the leftmost subplot shows y tick labels); every subplot still
    gets its own individual x-axis, but all subplots in the same row share
    the same fixed x/y range. One shared legend is drawn horizontally
    below the whole grid.
    """
    plt.rcParams.update({"font.size": FONT_SIZE})

    dataset_names = list(data.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 4 * 2), sharey="row")
    axes = axes.reshape(2, n)  # keep a consistent 2D shape even when n == 1

    row_axis_limits = (ROW_LIMITS, COL_LIMITS)

    for row_idx, experiment in enumerate(("row", "col")):
        xlim, ylim = row_axis_limits[row_idx]

        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axes[row_idx, col_idx]
            for algo, algo_data in data[dataset_name].items():
                series = algo_data[experiment]
                xs, ys = series["x"], series["y"]
                if experiment == "col":
                    xs, ys = xs[::2], ys[::2]
                ax.plot(xs, ys, marker="o", label=algo)

            if experiment == "row":
                ax.set_xscale("log")
                ax.set_xticks([10 ** e for e in range(3, 9)])
            else:
                ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, 30)))
                ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, 10)), minor=True)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(ROW_X_LABEL if experiment == "row" else COL_X_LABEL)
            ax.grid(True, which="major")
            if experiment == "col":
                ax.grid(True, which="minor", alpha=0.3)

            if col_idx != 0:
                ax.tick_params(labelleft=False)

        axes[row_idx, 0].set_ylabel(Y_LABEL)
        # a bold row header, further left than the (deliberately plain) y label
        axes[row_idx, 0].text(
            -0.4,
            0.5,
            ROW_SIDE_LABELS[row_idx],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )

    for col_idx, dataset_name in enumerate(dataset_names):
        axes[0, col_idx].set_title(dataset_name)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    fig.subplots_adjust(wspace=0.15, hspace=0.4, bottom=0.2)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def loadResults(path: str = RESULTS_PATH) -> SquareData:
    """Load results.pickle, shaped dataset -> algo -> {'row': (xs, ys), 'col': (xs, ys)}
    (xs/ys as arrays), and reshape it into plotSquare()'s expected
    dataset -> algo -> {'row': {'x': [...], 'y': [...]}, 'col': {'x': [...], 'y': [...]}}."""
    with open(path, "rb") as f:
        raw = pickle.load(f)

    data: SquareData = {}
    for dataset_name, algos in raw.items():
        dataset_name = DATASET_NAME_OVERRIDES.get(dataset_name, dataset_name)
        data[dataset_name] = {}
        for algo, experiments in algos.items():
            algo = ALGO_NAME_OVERRIDES.get(algo, algo)
            data[dataset_name][algo] = {
                experiment: {"x": list(xs), "y": list(ys)}
                for experiment, (xs, ys) in experiments.items()
            }
    return data


def _mockRowSeries(base: float, n: int = 6) -> Dict[str, List[float]]:
    """Log-spaced x in [1e3, 1e8], y growing with x but staying within [0, 300]."""
    log_min, log_max = math.log10(ROW_LIMITS[0][0]), math.log10(ROW_LIMITS[0][1])
    xs = [10 ** (log_min + i * (log_max - log_min) / (n - 1)) for i in range(n)]
    ys = [base * (x / ROW_LIMITS[0][0]) ** 0.3 for x in xs]
    return {"x": xs, "y": ys}


def _mockColSeries(base: float, n: int = 6) -> Dict[str, List[float]]:
    """Linear x in [10, 100], y growing with x but staying within [0, 300]."""
    x_min, x_max = COL_LIMITS[0]
    xs = [x_min + i * (x_max - x_min) / (n - 1) for i in range(n)]
    ys = [base * math.exp((x - x_min) / 30) for x in xs]
    return {"x": xs, "y": ys}



        
def main() -> None:
    data = loadResults(RESULTS_PATH)
    plotSquare(data)


if __name__ == "__main__":
    main()
