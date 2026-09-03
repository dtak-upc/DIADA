"""
Experiment 1: LIMA scalability vs. Apriori/FP-Growth baselines.

Live, re-runnable port of
experiment_results/experiment_1_scalability/benchmark.py -- the original,
pre-port script that produced
experiment_results/experiment_1_scalability/results.pickle, the data behind
the paper's published scalability figure
(experiment_results/experiment_1_scalability/squareplot.py). See
EXPERIMENTS.md at the repo root for the running list of numbered
reproducibility experiments this and future scripts belong to.

What this can rerun today vs. what it can't yet
-------------------------------------------------
  - LIMA ("ours"): fully rerunnable. LIMA / LIMAScheduler / LIMASampler /
    Dataset.expand() all already exist in the ported package with an API
    that matches the original script's usage exactly (verified directly:
    LIMA.__init__ accepts either a CSV path or an already-built Dataset,
    exactly as the original script relies on).
  - Apriori / FP-Growth: NOT rerunnable yet. The original script depends on
    APRIORISampler, FPSampler and MonoScheduler, none of which exist
    anywhere in the ported LIMA tree (nor in the pre-port LIMA_py source) --
    they were apparently never carried over from wherever LIMA itself was
    ported from. This script auto-detects whether they exist yet (see
    _try_import below) and only runs whichever algorithms are actually
    available. Once the missing sampler/scheduler files are added at the
    import paths noted below, this script picks them up automatically --
    no other change needed here.

Differences from the original
experiment_results/experiment_1_scalability/benchmark.py, all disclosed
here rather than silently applied
-----------------------------------------------------------------------
  1. Import paths fixed to the installed `diada.core.LIMA.*` package (the
     original used old flat, pre-port imports that no longer resolve).
  2. The original's main() computes a `rows_series` inside
     runScalabilityBenchmark() (via _rowsSeries()/_runSeries(vary_rows=True))
     for every (dataset, algo) pair, but then only ever reads the
     `col_series` half of what that function returns -- the rows_series
     values are computed and thrown away. This port skips computing them
     entirely (see run_column_series below): it does not change any output
     value, only the time needed to produce it.
  3. The "row" axis in results_live.pickle / the figure is NOT row-count
     growth (that would be _rowsSeries(), see point 2) -- it comes from a
     *different* function in the original, runSampleSizeScalabilityBenchmark(),
     which sweeps the sampling budget (approx = 1/aproxInv) on the
     ORIGINAL, un-expanded dataset. Confusing, but this port keeps that
     naming to match the original script and results.pickle's schema
     exactly (run_sample_size_series below is what fills the "row" key).
  4. The three datasets the original script names (miniboone.csv,
     jannis.csv, drive_diagnosis.csv) are pre-labelled DIADA/LIMA input
     files -- Dataset() infers a column's dtype from an "Integer"/"Double"
     substring in its *name*, so it needs those suffixes already present.
     Files matching that description no longer exist anywhere in either
     connected repo. What DOES exist is the *raw*, unlabelled version of
     each at data/base_datasets/{name}.csv (already part of this repo's
     standard benchmark suite, listed in benchmarks.yaml). This port
     regenerates the labelled input LIMA needs by running the same
     `_label_columns(df, num_buckets=1)` helper diada_tool.py uses ahead of
     every other DIADA/LIMA call in this package (Integer/Double/String
     typing, no bucketing -- bucketing every numeric column to String would
     change what's actually being benchmarked), caching the labelled CSV
     once per dataset under CACHE_DIR.
  5. RECONCILED (was previously flagged as unreconcilable -- it wasn't).
     benchmark.py's checked-in _colsSeries() (10..60 step 2) and
     APROX_INV_POINTS=40 do NOT match what actually produced results.pickle
     (both now living in experiment_results/experiment_1_scalability/).
     Reverse-engineered directly from the pickle's own x-values:
       - col series: 50 points, x = 0, 2, 4, ..., 98 -- i.e. range(0, 100, 2),
         not range(10, 61, 2). squareplot.py further clips its plot to
         COL_LIMITS=(10, 100) and subsamples with xs[::2] before drawing,
         which is why the *figure* looks like it spans roughly 10..96 --
         that's a plotting-time display artifact on top of the real 0..98
         data, not a different underlying range.
       - sample-size ("row") series: exactly 20 log-spaced points from
         1e3 to 1e8 (min/max match the checked-in APROX_INV_MIN/MAX; only
         the point count is wrong at 40). Cross-checked against the
         early-stopped FP-GROWTH series in the pickle (17/19 points on two
         datasets), whose cutoff values land exactly on the 17th/19th term
         of this same 20-point log sequence -- confirms 20, not 40.
     _cols_series() and APROX_INV_POINTS below now use these recovered
     values so a live rerun lands on the same axes as the published figure.
  6. Resumable: every (dataset, algo, series_type, x) timing point is
     checkpointed to scalability_raw.csv before moving to the next point,
     so a long sweep survives being interrupted and restarted. This is
     expected to take a long time: REPEATS=5 timed repetitions per point,
     up to 26 column points + 40 sample-size points per (dataset, algo),
     each REPEATS-averaged run itself capped by TIME_LIMIT_SECONDS.

Run with: python3 -m diada.experiments.experiment_1_scalability.scalability_benchmark
Prints progress continuously; prints "ALL DONE" at the very end.
Env vars (both optional): OUT_DIR (default results/experiment_1_scalability
under the repo root), CACHE_DIR (default OUT_DIR/cache).
"""
from __future__ import annotations

import importlib
import math
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ... import paths
from ...core.diada_tool import _label_columns
from ...core.LIMA.data.dataset.Dataset import Dataset
from ...core.LIMA.LIMA.LIMA import LIMA
from ...core.LIMA.LIMA.sampler.LIMASampler import LIMASampler
from ...core.LIMA.LIMA.scheduler.LIMAScheduler import LIMAScheduler


def _try_import(module_path: str, attr: str):
    try:
        return getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError):
        return None


# Optional baselines -- see module docstring. Left as None (and skipped)
# until these files exist in the ported package.
APRIORISampler = _try_import("diada.core.LIMA.LIMA.sampler.APRIORISampler", "APRIORISampler")
FPSampler = _try_import("diada.core.LIMA.LIMA.sampler.FPSampler", "FPSampler")
MonoScheduler = _try_import("diada.core.LIMA.LIMA.scheduler.MonoScheduler", "MonoScheduler")

_HAVE_APRIORI = APRIORISampler is not None and MonoScheduler is not None
_HAVE_FPGROWTH = FPSampler is not None and MonoScheduler is not None

ALL_ALGORITHMS = ("apriori", "FP-GROWTH", "LIMA")
ALGORITHMS = tuple(
    a for a in ALL_ALGORITHMS
    if a == "LIMA" or (a == "apriori" and _HAVE_APRIORI) or (a == "FP-GROWTH" and _HAVE_FPGROWTH)
)

DATASET_NAMES = ["miniboone", "jannis", "drive_diagnosis"]
APRIORI_FP_DEPTH = 3
REPEATS = 5
COLUMN_SCALABILITY_APPROX = 1e-5
TIME_LIMIT_SECONDS = 1800.0

APROX_INV_MIN = 1e3
APROX_INV_MAX = 1e8
APROX_INV_POINTS = 20  # recovered from results.pickle -- see docstring point 5

_BASE_ROWS = 20_000  # rows held fixed while the columns series grows


def _cols_series() -> List[int]:
    # recovered from results.pickle -- see module docstring point 5
    return list(range(0, 100, 2))


def _aprox_inv_series() -> List[float]:
    if APROX_INV_POINTS == 1:
        return [APROX_INV_MIN]
    log_min, log_max = math.log10(APROX_INV_MIN), math.log10(APROX_INV_MAX)
    step = (log_max - log_min) / (APROX_INV_POINTS - 1)
    return [10 ** (log_min + i * step) for i in range(APROX_INV_POINTS)]


OUT_DIR = Path(os.environ.get("OUT_DIR", str(paths.RESULTS_DIR / "experiment_1_scalability")))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(OUT_DIR / "cache")))
RAW_CSV = OUT_DIR / "scalability_raw.csv"
PICKLE_PATH = OUT_DIR / "results_live.pickle"
PLOT_PATH = OUT_DIR / "scalability_live.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_or_label_dataset(name: str) -> Dataset:
    labelled_path = CACHE_DIR / f"{name}_labelled.csv"
    if not labelled_path.exists():
        log(f"[{name}] labelling raw base dataset (Integer/Double/String typing, no bucketing)...")
        raw = pd.read_csv(paths.BASE_DATASETS_DIR / f"{name}.csv")
        labelled = _label_columns(raw, num_buckets=1)
        labelled.to_csv(labelled_path, index=False)
    return Dataset(str(labelled_path))


def _raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame(columns=["series_type", "dataset", "algo", "x", "elapsed_seconds"])
    return pd.read_csv(RAW_CSV)


def already_done(series_type: str, dataset: str, algo: str, x) -> bool:
    done = _raw_df()
    m = ((done["series_type"] == series_type) & (done["dataset"] == dataset)
         & (done["algo"] == algo) & (done["x"].astype(float) == float(x)))
    return m.any()


def series_stopped(series_type: str, dataset: str, algo: str) -> bool:
    """True once some point already run in this series exceeded
    TIME_LIMIT_SECONDS -- mirrors the original's early-stop behavior across
    resumed invocations of this script."""
    done = _raw_df()
    m = ((done["series_type"] == series_type) & (done["dataset"] == dataset) & (done["algo"] == algo))
    sub = done[m]
    return bool((sub["elapsed_seconds"] > TIME_LIMIT_SECONDS).any())


def append_point(row):
    df = pd.DataFrame([row])
    header = not RAW_CSV.exists()
    df.to_csv(RAW_CSV, mode="a" if not header else "w", header=header, index=False)


def _derive_params(approx: float, dataset_size: int) -> Tuple[int, int]:
    estN = min(10.0 / approx, 1 * dataset_size ** 2)
    return int(estN), 100


def _time_run(lima: LIMA, max_iter: int) -> float:
    start = time.perf_counter()
    for i in range(max_iter):
        if lima.step(i):
            break
    return time.perf_counter() - start


def _build_lima(dataset: Dataset, approx: float, algo: str, dataset_size: int) -> LIMA:
    if algo == "LIMA":
        return LIMA(dataset, scheduler=LIMAScheduler(), sampler=LIMASampler(), approx=approx)
    estN, _ = _derive_params(approx, dataset_size)
    if algo == "apriori":
        sampler = APRIORISampler(APRIORI_FP_DEPTH)
    elif algo == "FP-GROWTH":
        sampler = FPSampler(APRIORI_FP_DEPTH)
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    return LIMA(dataset, scheduler=MonoScheduler(), sampler=sampler, approx=approx)


def _run_algo(dataset: Dataset, approx: float, algo: str) -> float:
    dataset_size = len(dataset.df)
    _, max_iter = _derive_params(approx, dataset_size)
    elapsed_times = []
    for _ in range(REPEATS):
        lima = _build_lima(dataset, approx, algo, dataset_size)
        elapsed_times.append(_time_run(lima, max_iter))
    return sum(elapsed_times) / len(elapsed_times)


def run_column_series(name: str, base_dataset: Dataset, algo: str):
    """Column-count growth, rows fixed at _BASE_ROWS -- fills the 'col' key."""
    for cols in _cols_series():
        if already_done("col", name, algo, cols):
            continue
        if series_stopped("col", name, algo):
            log(f"[{name} | {algo} | col] series already stopped (a prior point exceeded "
                f"{TIME_LIMIT_SECONDS:.0f}s) -- skipping cols={cols}")
            continue
        log(f"[{name} | {algo} | col] rows={_BASE_ROWS} cols={cols} running ({REPEATS} repeats)...")
        expanded = base_dataset.expand(_BASE_ROWS, cols)
        elapsed = _run_algo(expanded, COLUMN_SCALABILITY_APPROX, algo)
        append_point({"series_type": "col", "dataset": name, "algo": algo, "x": cols, "elapsed_seconds": elapsed})
        log(f"[{name} | {algo} | col] cols={cols} done in {elapsed:.3f}s")


def run_sample_size_series(name: str, base_dataset: Dataset, algo: str):
    """Sampling-budget growth on the ORIGINAL dataset -- fills the 'row' key
    (see module docstring point 3 for why it's called 'row')."""
    for aprox_inv in _aprox_inv_series():
        if already_done("row", name, algo, aprox_inv):
            continue
        if series_stopped("row", name, algo):
            log(f"[{name} | {algo} | row] series already stopped -- skipping aproxInv={aprox_inv:.3g}")
            continue
        approx = 1.0 / aprox_inv
        log(f"[{name} | {algo} | row] aproxInv={aprox_inv:.3g} (approx={approx:.3g}) running ({REPEATS} repeats)...")
        elapsed = _run_algo(base_dataset, approx, algo)
        append_point({"series_type": "row", "dataset": name, "algo": algo, "x": aprox_inv, "elapsed_seconds": elapsed})
        log(f"[{name} | {algo} | row] aproxInv={aprox_inv:.3g} done in {elapsed:.3f}s")


_ALGO_COLORS = {"LIMA": "#2a78d6", "apriori": "#eb6834", "FP-GROWTH": "#1baf7a"}
_ALGO_LABELS = {"LIMA": "LIMA (ours)", "apriori": "Apriori", "FP-GROWTH": "FP-Growth"}


def _plot(combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in DATASET_NAMES if combined.get(n)]
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 8), squeeze=False)
    for col_idx, name in enumerate(names):
        for row_idx, series_type in enumerate(("row", "col")):
            ax = axes[row_idx][col_idx]
            for algo, series in combined[name].items():
                xs, ys = series[series_type]
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", color=_ALGO_COLORS.get(algo, "#52514e"),
                        label=_ALGO_LABELS.get(algo, algo))
            if series_type == "row":
                ax.set_xscale("log")
                ax.set_xlabel(r"$\epsilon^{-1}$ (sample-size budget)")
            else:
                ax.set_xlabel("columns")
            if col_idx == 0:
                ax.set_ylabel(f"{'row' if series_type == 'row' else 'column'} scalability\ntime (s)")
            ax.grid(True, alpha=0.3)
        axes[0][col_idx].set_title(name)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.subplots_adjust(bottom=0.15, hspace=0.35)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {PLOT_PATH}")


def build_outputs():
    """Reshape scalability_raw.csv into results_live.pickle (same schema as
    experiment_results/experiment_1_scalability/results.pickle: dataset -> algo -> {'row': (xs, ys),
    'col': (xs, ys)}) and render scalability_live.png."""
    raw = _raw_df()
    if raw.empty:
        log("no data points yet, skipping output build")
        return
    combined = {}
    for name in DATASET_NAMES:
        per_algo = {}
        for algo in ALGORITHMS:
            algo_rows = raw[(raw["dataset"] == name) & (raw["algo"] == algo)]
            if algo_rows.empty:
                continue
            series = {}
            for series_type in ("row", "col"):
                sub = algo_rows[algo_rows["series_type"] == series_type].sort_values("x")
                series[series_type] = (sub["x"].tolist(), sub["elapsed_seconds"].tolist())
            per_algo[algo] = series
        if per_algo:
            combined[name] = per_algo

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(combined, f)
    log(f"wrote {PICKLE_PATH}")
    _plot(combined)


def main():
    log(f"Experiment 1 (scalability) -- available algorithms: {ALGORITHMS}")
    if not _HAVE_APRIORI:
        log("  apriori: NOT AVAILABLE (APRIORISampler and/or MonoScheduler not found under "
            "diada.core.LIMA.LIMA.* -- skipping until added)")
    if not _HAVE_FPGROWTH:
        log("  FP-GROWTH: NOT AVAILABLE (FPSampler and/or MonoScheduler not found under "
            "diada.core.LIMA.LIMA.* -- skipping until added)")

    for name in DATASET_NAMES:
        base_dataset = get_or_label_dataset(name)
        for algo in ALGORITHMS:
            run_sample_size_series(name, base_dataset, algo)
            run_column_series(name, base_dataset, algo)

    build_outputs()
    log("ALL DONE")


if __name__ == "__main__":
    main()
