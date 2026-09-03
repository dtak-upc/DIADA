"""
Shared plotting/analysis helpers for the ARDA feature-count-sweep result
tables -- Experiment 3's results_final_*.csv and Experiment 4's
results_2targets_*.csv.

Extracted out of results/results.ipynb, where this logic was duplicated
almost verbatim between the two experiments' analysis cells -- only the
`variants` dict (which suffix/color/label goes with which benchmark
variant), the noisy-variant suffix marker, and a couple of dataset-ordering
constants differed. No behavior changes versus the notebook version, only
parameterizing what used to be hardcoded module-level globals so both
experiments (and any future one with the same results_final_*.csv shape)
can call the same functions with their own config instead of duplicating
the whole thing again. One small, deliberate generalization: the original
checked `"Base" not in cfg["label"]` in Experiment 3's copy vs.
`"base" not in cfg["label"]` in Experiment 4's -- unified here as a
case-insensitive check (`"base" not in cfg["label"].lower()`), which
matches both.

Used by:
  - experiment_results/experiment_3_diada_arda/results_analysis.ipynb
  - experiment_results/experiment_4_synthetic_targets/results_analysis.ipynb
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

DEFAULT_NOISY_PATTERNS = ["noise_", "_shuffled_", "spurious_"]
DEFAULT_COLUMN_COLORS = ["#ffffff", "#fff5e6"]
DEFAULT_STYLE = {
    "title_size": 14, "label_size": 12, "tick_size": 11, "legend_size": 11,
    "pair_hspace": 0.1, "group_hspace": -0.08, "fig_w": 3.5, "fig_h": 3, "pair_wspace": 0.3,
}


def compute_noisy_percentage(selected_features_str, noisy_patterns: List[str] = DEFAULT_NOISY_PATTERNS) -> float:
    if pd.isna(selected_features_str) or selected_features_str == "":
        return np.nan
    features = [f.strip() for f in selected_features_str.split(",")]
    if len(features) == 0:
        return np.nan
    noisy_count = sum(any(p in feat for p in noisy_patterns) for feat in features)
    return 100.0 * noisy_count / len(features)


def get_base_benchmark(name: str, suffixes: List[str]) -> str:
    for s in suffixes:
        if s and name.endswith(s):
            return name[: -len(s)]
    return name


def plot_dataset(
    df: pd.DataFrame,
    task_type: str,
    metric_col: str,
    metric_label: str,
    variants: Dict[str, dict],
    max_features_per_benchmark: Dict[str, int],
    noisy_patterns: List[str] = DEFAULT_NOISY_PATTERNS,
    style: Dict[str, float] = DEFAULT_STYLE,
    column_colors: List[str] = DEFAULT_COLUMN_COLORS,
    classification_priority_dataset: Optional[str] = None,
    classification_excluded_datasets: Optional[List[str]] = None,
    ncols: int = 7,
):
    """One figure per task_type: for every base benchmark, a top subplot
    (metric_col vs. num_new_features, one line per variant) stacked over a
    bottom subplot (% of selected features that are noise, same x-axis).
    `variants` = {name: {"suffix", "color", "label", "linestyle", "marker"}}.
    The classification-only dataset reordering/exclusion (mirrors the
    original notebook, which only applied this for task_type ==
    "classification") is opt-in via the two `classification_*` params."""
    import matplotlib.pyplot as plt

    df = df.copy()
    df["noisy_percentage"] = df["selected_features"].apply(lambda s: compute_noisy_percentage(s, noisy_patterns))

    suffixes = [cfg["suffix"] for cfg in variants.values()]
    base_benchmarks = sorted(set(get_base_benchmark(b, suffixes) for b in df["benchmark"].unique()))

    if task_type == "classification":
        for dataset in classification_excluded_datasets or []:
            if dataset in base_benchmarks:
                base_benchmarks.remove(dataset)
        if classification_priority_dataset and classification_priority_dataset in base_benchmarks:
            base_benchmarks = [classification_priority_dataset] + [
                b for b in base_benchmarks if b != classification_priority_dataset
            ]

    nrows = math.ceil(len(base_benchmarks) / ncols)
    fig = plt.figure(figsize=(style["fig_w"] * ncols, style["fig_h"] * nrows * 1.5))
    subfigs = fig.subfigures(nrows, 1, hspace=style["group_hspace"])
    if nrows == 1:
        subfigs = [subfigs]

    for row_idx, subfig in enumerate(subfigs):
        if row_idx == 0:
            subfig.text(0.5, 0.95, "Classification tasks" if task_type == "classification" else "Regression tasks",
                        ha="center", va="bottom", fontsize=20, fontweight="bold")
        subfig.patch.set_alpha(0.55)
        axes_grid = subfig.subplots(2, ncols, sharex=False,
                                     gridspec_kw={"hspace": style["pair_hspace"], "wspace": style["pair_wspace"]})
        if ncols == 1:
            axes_grid = axes_grid.reshape(2, 1)

        for col_idx in range(ncols):
            bench_idx = row_idx * ncols + col_idx
            if bench_idx >= len(base_benchmarks):
                subfig.delaxes(axes_grid[0, col_idx])
                subfig.delaxes(axes_grid[1, col_idx])
                continue

            base = base_benchmarks[bench_idx]
            ax_top, ax_bottom = axes_grid[0, col_idx], axes_grid[1, col_idx]

            bg_color = column_colors[col_idx % len(column_colors)]
            ax_top.set_facecolor(bg_color)
            ax_bottom.set_facecolor(bg_color)
            ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

            for _, cfg in variants.items():
                subset = df[df["benchmark"] == base + cfg["suffix"]]
                if subset.empty:
                    continue

                grouped = (
                    subset.groupby("num_new_features")
                    .agg({metric_col: "mean", "noisy_percentage": "mean"})
                    .reset_index()
                    .sort_values("num_new_features")
                )

                max_features = max_features_per_benchmark.get(base)
                if max_features is not None:
                    grouped = grouped[grouped["num_new_features"] <= max_features]

                ax_top.plot(grouped["num_new_features"], grouped[metric_col],
                            linestyle=cfg["linestyle"], color=cfg["color"], label=cfg["label"],
                            linewidth=3, markevery=4)

                if "base" not in cfg["label"].lower():
                    ax_bottom.plot(grouped["num_new_features"], grouped["noisy_percentage"],
                                   linestyle=cfg["linestyle"], color=cfg["color"], linewidth=3, markevery=4)

                if max_features is not None:
                    ax_top.set_xlim(0, max_features + 1)
                    ax_bottom.set_xlim(0, max_features + 1)

                ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_bottom.set_ylim(-2, 100)
                ax_bottom.set_yticks(np.arange(0, 101, 20))

            clean_base = base.replace("_", " ").capitalize()
            ax_top.set_title(clean_base, fontsize=style["title_size"], fontweight="bold")

            if col_idx == 0:
                ax_top.set_ylabel(metric_label, fontsize=style["label_size"])
                ax_bottom.set_ylabel("Noisy features (%)", fontsize=style["label_size"])
            ax_bottom.set_xlabel("Number of features", fontsize=style["label_size"])

            ax_top.tick_params(axis="both", labelsize=style["tick_size"])
            ax_bottom.tick_params(axis="both", labelsize=style["tick_size"])
            ax_top.grid(True, alpha=0.3)
            ax_bottom.grid(True, alpha=0.3)

            if bench_idx == 0:
                ax_top.legend(fontsize=style["legend_size"])

    plt.tight_layout()
    return fig


def count_noise_types(selected_features_str, noisy_patterns: List[str] = DEFAULT_NOISY_PATTERNS) -> Dict[str, int]:
    counts = {p: 0 for p in noisy_patterns}
    if pd.isna(selected_features_str) or selected_features_str == "":
        return counts
    features = [f.strip() for f in selected_features_str.split(",")]
    for feat in features:
        for p in noisy_patterns:
            if p in feat:
                counts[p] += 1
    return counts


def print_noise_table(df: pd.DataFrame, dirty_marker: str = "_dirty",
                       noisy_patterns: List[str] = DEFAULT_NOISY_PATTERNS) -> pd.DataFrame:
    """For every noisy (uncleaned) benchmark variant -- i.e. whose name
    contains `dirty_marker` -- sum how many selected-feature slots are each
    noise type, at the largest num_new_features tested. `dirty_marker`
    selects which benchmark-name substring marks an uncleaned variant
    (default matches Experiment 3's `_un_dirty`/`_mn_dirty`; pass the
    equivalent marker for a different variant naming scheme)."""
    rows = []
    for benchmark in sorted(df["benchmark"].unique()):
        if dirty_marker not in benchmark:
            continue
        subset = df[df["benchmark"] == benchmark]
        last_k = subset["num_new_features"].max()
        last_subset = subset[subset["num_new_features"] == last_k]

        total_counts = {p: 0 for p in noisy_patterns}
        for features in last_subset["selected_features"]:
            counts = count_noise_types(features, noisy_patterns)
            for p in noisy_patterns:
                total_counts[p] += counts[p]

        row = {"benchmark": benchmark, **total_counts, "total_noisy": sum(total_counts.values())}
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\nNOISE TYPE COUNTS")
    print(table.to_string(index=False))
    return table


def plot_augmentation_time_bars(df: pd.DataFrame, variants: Dict[str, dict], ncols: int = 3):
    """Grouped bar chart of mean `augmentation_time` per base benchmark,
    one bar per variant. `df` must have `benchmark` and `augmentation_time`
    columns (e.g. results_final_classification.csv + _regression.csv,
    concatenated and deduplicated)."""
    import matplotlib.pyplot as plt

    suffixes = [cfg["suffix"] for cfg in variants.values()]
    base_benchmarks = sorted(set(get_base_benchmark(b, suffixes) for b in df["benchmark"].unique()))

    n_benchmarks = len(base_benchmarks)
    nrows = math.ceil(n_benchmarks / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, base in enumerate(base_benchmarks):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        means, colors, labels = [], [], []
        for _, cfg in variants.items():
            subset = df[df["benchmark"] == base + cfg["suffix"]]
            if subset.empty:
                continue
            means.append(subset["augmentation_time"].mean())
            colors.append(cfg["color"])
            labels.append(cfg.get("bar_label", cfg["label"]))

        x = np.arange(len(means))
        ax.bar(x, means, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(base)
        ax.set_ylabel("augmentation_time")
        ax.grid(True, axis="y", alpha=0.3)

    for empty_idx in range(n_benchmarks, nrows * ncols):
        row, col = empty_idx // ncols, empty_idx % ncols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    return fig
