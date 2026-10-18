"""
Experiment 1: LIMA scalability vs. Apriori/FP-Growth baselines.

Live, re-runnable port of
experiment_results/experiment_1_scalability/benchmark.py -- the original,
pre-port script that produced
experiment_results/experiment_1_scalability/results.pickle, the data behind
the paper's published scalability figure (Figure 4, drawn by
experiment_results/experiment_1_scalability/squareplot.py).

What it runs
------------
For each dataset (miniboone, jannis, drive_diagnosis) and each algorithm
(LIMA, Apriori, FP-Growth), two timing series:
  - "row" (row scalability): the sampling budget eps^-1 swept over 20
    log-spaced points from 1e3 to 1e8, on the original dataset.
  - "col" (column scalability): eps fixed at 1e-5, rows fixed at 20,000,
    and the column count swept over range(0, 100, 2) via Dataset.expand().
Every point is the mean of REPEATS (5) timed runs.

Algorithms:
  - LIMA ("ours", DIADA in the paper): LIMAScheduler + LIMASampler.
  - Apriori / FP-Growth: MonoScheduler + APRIORISampler / FPSampler. These
    sample one fixed domain of n = BASELINE_SAMPLE_K / eps pairs and
    enumerate every predicate set of up to APRIORI_FP_DEPTH (3) predicates.

Known caveat: on this port, LIMA's timings do NOT match Figure 4 / the
pickle (roughly 10x slower on miniboone at 20 columns, eps=1e-5). The
baselines run correctly: on a shared sample, Apriori and FP-Growth return
identical counts for every lattice node. So this script runs the
experiment as described in the paper, but it is not expected to reproduce
the published numbers until the LIMA port is fixed.

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
  5. benchmark.py's checked-in _colsSeries() (10..60 step 2) and
     APROX_INV_POINTS=40 do NOT match what actually produced results.pickle.
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
  6. Baseline sample size. The paper (Section 6.1, footnote 5) sets
     |Omega'| = k / eps with k = 5. The original computed 2 * min(10/eps, N^2)
     = 20/eps but never passed it to LIMA.step() (which only takes the
     iteration number), so its baselines always sampled 1024 pairs whatever
     eps was. This port uses BASELINE_SAMPLE_K = 5 and actually applies it
     (see _build_lima).
  7. Column draws. Figure 4's caption says each of the 5 repetitions
     evaluates "different random sets of columns"; the original expanded
     once per point and reused it for all repeats. This port redraws the
     expansion for every repeat of the column series.
  8. Guards. A point's remaining repeats are skipped as soon as one repeat
     exceeds the time limit (the series then stops, as in the original), and
     a MemoryError is recorded as status "oom" and stops the series -- the
     figure leaves those entries missing, as the paper's "Missing entries
     indicate memory overflows" describes. A repeat already running is
     never interrupted, so a single point can still overrun the limit.
  9. Resumable: every (dataset, algo, series_type, x) timing point is
     checkpointed to scalability_raw.csv before moving to the next point,
     so a long sweep survives being interrupted and restarted. A full run
     is expected to take a long time (days, not hours).

Usage
-----
Full experiment (resumable):
    python -m diada.experiments.experiment_1_scalability.scalability_benchmark
Quick smoke run (1 repeat, a handful of small points, separate output dir):
    python -m diada.experiments.experiment_1_scalability.scalability_benchmark --quick
Subsets / overrides, e.g.:
    ... --datasets jannis --algorithms LIMA apriori --series col --max-cols 30 --repeats 2
Only re-render the figure from an existing scalability_raw.csv:
    ... --plot-only
Run with --help for every option. Prints progress continuously and
"ALL DONE" at the very end. Outputs (under --out-dir, default
results/experiment_1_scalability; env var OUT_DIR also works):
scalability_raw.csv, results_live.pickle (same schema as results.pickle),
scalability_live.png (Figure 4 layout). Labelled inputs are cached under
--cache-dir (default <out-dir>/cache; env var CACHE_DIR).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ... import paths
from ...core.diada_tool import _label_columns
from ...core.LIMA.data.dataset.Dataset import Dataset
from ...core.LIMA.LIMA.LIMA import LIMA
from ...core.LIMA.LIMA.sampler.APRIORISampler import APRIORISampler
from ...core.LIMA.LIMA.sampler.FPSampler import FPSampler
from ...core.LIMA.LIMA.sampler.LIMASampler import LIMASampler
from ...core.LIMA.LIMA.scheduler.LIMAScheduler import LIMAScheduler
from ...core.LIMA.LIMA.scheduler.MonoScheduler import MonoScheduler

ALL_ALGORITHMS = ("apriori", "FP-GROWTH", "LIMA")
ALL_DATASETS = ("miniboone", "jannis", "drive_diagnosis")
ALL_SERIES = ("row", "col")

APRIORI_FP_DEPTH = 3
COLUMN_SCALABILITY_APPROX = 1e-5

# See module docstring point 6.
BASELINE_SAMPLE_K = 5.0

APROX_INV_MIN = 1e3
APROX_INV_MAX = 1e8

_BASE_ROWS = 20_000  # rows held fixed while the columns series grows

# Run configuration -- defaults reproduce the paper's protocol; main()
# overrides them from the command line.
DATASET_NAMES: List[str] = list(ALL_DATASETS)
ALGORITHMS: Tuple[str, ...] = ALL_ALGORITHMS
SERIES: Tuple[str, ...] = ALL_SERIES
REPEATS = 5
TIME_LIMIT_SECONDS = 1800.0
APROX_INV_POINTS = 20  # recovered from results.pickle -- see docstring point 5
ROW_APROX_INV_MAX = APROX_INV_MAX  # last eps^-1 point actually run (--row-max)
ROW_COLS: Optional[int] = None  # None = row series on the original dataset (paper); else a fixed random expansion
MAX_COLS = 98  # last point of range(0, 100, 2), see docstring point 5
PAPER_AXES = True  # Figure 4 axis limits; main() turns it off for reduced runs


def _cols_series() -> List[int]:
    # recovered from results.pickle -- see module docstring point 5
    return [c for c in range(0, 100, 2) if c <= MAX_COLS]


def _aprox_inv_series() -> List[float]:
    if APROX_INV_POINTS == 1:
        return [APROX_INV_MIN]
    log_min, log_max = math.log10(APROX_INV_MIN), math.log10(ROW_APROX_INV_MAX)
    step = (log_max - log_min) / (APROX_INV_POINTS - 1)
    return [10 ** (log_min + i * step) for i in range(APROX_INV_POINTS)]


OUT_DIR: Path
CACHE_DIR: Path
RAW_CSV: Path
PICKLE_PATH: Path
PLOT_PATH: Path


def _set_out_dir(out_dir: Path, cache_dir: Optional[Path] = None) -> None:
    global OUT_DIR, CACHE_DIR, RAW_CSV, PICKLE_PATH, PLOT_PATH
    OUT_DIR = Path(out_dir)
    CACHE_DIR = Path(cache_dir) if cache_dir is not None else OUT_DIR / "cache"
    RAW_CSV = OUT_DIR / "scalability_raw.csv"
    PICKLE_PATH = OUT_DIR / "results_live.pickle"
    PLOT_PATH = OUT_DIR / "scalability_live.png"


_DEFAULT_OUT_DIR = Path(os.environ.get("OUT_DIR", str(paths.RESULTS_DIR / "experiment_1_scalability")))
_set_out_dir(_DEFAULT_OUT_DIR, Path(os.environ["CACHE_DIR"]) if "CACHE_DIR" in os.environ else None)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_or_label_dataset(name: str) -> Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labelled_path = CACHE_DIR / f"{name}_labelled.csv"
    if not labelled_path.exists():
        log(f"[{name}] labelling raw base dataset (Integer/Double/String typing, no bucketing)...")
        raw = pd.read_csv(paths.BASE_DATASETS_DIR / f"{name}.csv")
        labelled = _label_columns(raw, num_buckets=1)
        labelled.to_csv(labelled_path, index=False)
    return Dataset(str(labelled_path))


_RAW_COLUMNS = ["series_type", "dataset", "algo", "x", "elapsed_seconds", "repeats_run", "status"]


def _raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame(columns=_RAW_COLUMNS)
    return pd.read_csv(RAW_CSV)


def _series_rows(done: pd.DataFrame, series_type: str, dataset: str, algo: str) -> pd.DataFrame:
    return done[(done["series_type"] == series_type) & (done["dataset"] == dataset) & (done["algo"] == algo)]


def already_done(series_type: str, dataset: str, algo: str, x) -> bool:
    sub = _series_rows(_raw_df(), series_type, dataset, algo)
    return bool((sub["x"].astype(float) == float(x)).any())


def series_stopped(series_type: str, dataset: str, algo: str) -> bool:
    """True once some point already run in this series exceeded the time
    limit or ran out of memory -- mirrors the original's early-stop behavior
    across resumed invocations of this script."""
    sub = _series_rows(_raw_df(), series_type, dataset, algo)
    return bool(((sub["elapsed_seconds"] > TIME_LIMIT_SECONDS) | (sub["status"] != "ok")).any())


def append_point(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=_RAW_COLUMNS)
    header = not RAW_CSV.exists()
    df.to_csv(RAW_CSV, mode="w" if header else "a", header=header, index=False)


def _derive_params(approx: float, dataset_size: int) -> Tuple[int, int]:
    estN = min(BASELINE_SAMPLE_K / approx, 1 * dataset_size ** 2)
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
    lima = LIMA(dataset, scheduler=MonoScheduler(), sampler=sampler, approx=approx)
    # MonoScheduler samples exactly once; make that single step draw the
    # eps-derived sample size instead of LIMA's default 2**10 first step.
    lima.getStepSize = lambda iteration: estN
    return lima


def _run_point(make_dataset, approx: float, algo: str) -> Tuple[float, int, str]:
    """Time `algo` REPEATS times, each on `make_dataset()`, and return
    (mean elapsed seconds, repeats actually run, status). Stops repeating
    once a repeat exceeds TIME_LIMIT_SECONDS (status "timeout"); a
    MemoryError ends the point with status "oom" and NaN time."""
    elapsed_times = []
    for _ in range(REPEATS):
        try:
            dataset = make_dataset()
            dataset_size = len(dataset.df)
            _, max_iter = _derive_params(approx, dataset_size)
            lima = _build_lima(dataset, approx, algo, dataset_size)
            elapsed = _time_run(lima, max_iter)
        except MemoryError:
            return float("nan"), len(elapsed_times) + 1, "oom"
        elapsed_times.append(elapsed)
        if elapsed > TIME_LIMIT_SECONDS:
            return sum(elapsed_times) / len(elapsed_times), len(elapsed_times), "timeout"
    return sum(elapsed_times) / len(elapsed_times), len(elapsed_times), "ok"


def _run_series(name: str, algo: str, series_type: str, xs: Sequence, make_dataset_for, approx_for, fmt):
    for x in xs:
        tag = f"[{name} | {algo} | {series_type}] {fmt(x)}"
        if already_done(series_type, name, algo, x):
            continue
        if series_stopped(series_type, name, algo):
            log(f"{tag}: series already stopped (time limit / memory) -- skipping")
            continue
        log(f"{tag} running (up to {REPEATS} repeats)...")
        elapsed, repeats_run, status = _run_point(make_dataset_for(x), approx_for(x), algo)
        append_point({"series_type": series_type, "dataset": name, "algo": algo, "x": x,
                      "elapsed_seconds": elapsed, "repeats_run": repeats_run, "status": status})
        log(f"{tag} done in {elapsed:.3f}s ({status}, {repeats_run} repeat(s))")


def run_column_series(name: str, base_dataset: Dataset, algo: str):
    """Column-count growth, rows fixed at _BASE_ROWS -- fills the 'col' key.
    Every repeat draws a fresh random set of columns (docstring point 7)."""
    _run_series(
        name, algo, "col", _cols_series(),
        make_dataset_for=lambda cols: (lambda: base_dataset.expand(_BASE_ROWS, cols)),
        approx_for=lambda cols: COLUMN_SCALABILITY_APPROX,
        fmt=lambda cols: f"rows={_BASE_ROWS} cols={cols}",
    )


def run_sample_size_series(name: str, base_dataset: Dataset, algo: str):
    """Sampling-budget growth on the ORIGINAL dataset -- fills the 'row' key
    (see module docstring point 3 for why it's called 'row'). With
    --row-cols, a single seeded random expansion to that many columns (same
    rows) stands in for the original dataset at every point, for quick runs."""
    if ROW_COLS is not None:
        import numpy as np
        state = np.random.get_state()
        np.random.seed(0)
        base_dataset = base_dataset.expand(len(base_dataset.df), ROW_COLS)
        np.random.set_state(state)
    _run_series(
        name, algo, "row", _aprox_inv_series(),
        make_dataset_for=lambda aprox_inv: (lambda: base_dataset),
        approx_for=lambda aprox_inv: 1.0 / aprox_inv,
        fmt=lambda aprox_inv: f"aproxInv={aprox_inv:.3g} (approx={1.0 / aprox_inv:.3g})",
    )


# Figure 4 styling, matching experiment_results/experiment_1_scalability/squareplot.py
_ALGO_ORDER = ("LIMA", "apriori", "FP-GROWTH")
_ALGO_LABELS = {"LIMA": "DIADA", "apriori": "APRIORI", "FP-GROWTH": "FP-GROWTH"}
_ALGO_COLORS = {"LIMA": "#1f77b4", "apriori": "#ff7f0e", "FP-GROWTH": "#2ca02c"}
_DATASET_TITLES = {"miniboone": "miniboone", "jannis": "jannis", "drive_diagnosis": "drive diagnosis"}
_ROW_LIMITS = ((APROX_INV_MIN, APROX_INV_MAX), (0, 300))
_COL_LIMITS = ((10, 100), (0, 300))


def _plot(combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ALL_DATASETS if combined.get(n)]
    if not names:
        return
    plt.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 8), sharey="row", squeeze=False)
    for col_idx, name in enumerate(names):
        for row_idx, series_type in enumerate(("row", "col")):
            ax = axes[row_idx][col_idx]
            (xlim, ylim) = _ROW_LIMITS if series_type == "row" else _COL_LIMITS
            for algo in _ALGO_ORDER:
                if algo not in combined[name]:
                    continue
                xs, ys = combined[name][algo][series_type]
                if not xs:
                    continue
                if series_type == "col" and PAPER_AXES:
                    xs, ys = xs[::2], ys[::2]  # as squareplot.py does
                ax.plot(xs, ys, marker="o", color=_ALGO_COLORS[algo], label=_ALGO_LABELS[algo])
            if series_type == "row":
                ax.set_xscale("log")
                ax.set_xlabel(r"$\epsilon^{-1}$")
                if PAPER_AXES:
                    ax.set_xticks([10 ** e for e in range(3, 9)])
            else:
                ax.set_xlabel("columns")
                if PAPER_AXES:
                    ax.set_xticks(list(range(10, 101, 30)))
                    ax.set_xticks(list(range(10, 101, 10)), minor=True)
                    ax.grid(True, which="minor", alpha=0.3)
            if PAPER_AXES:  # reduced runs keep matplotlib's automatic limits
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            ax.grid(True, which="major")
        axes[0][col_idx].set_title(_DATASET_TITLES.get(name, name))
    for row_idx, label in enumerate(("row scalability", "column scalability")):
        axes[row_idx][0].set_ylabel("time (s)")
        axes[row_idx][0].text(-0.4, 0.5, label, transform=axes[row_idx][0].transAxes,
                              rotation=90, ha="center", va="center", fontweight="bold")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.subplots_adjust(wspace=0.15, hspace=0.4, bottom=0.2)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    log(f"wrote {PLOT_PATH}")


_PICKLE_DATASET_KEYS = {"drive_diagnosis": "drive"}
_PICKLE_ALGO_KEYS = {"apriori": "APRIORI"}


def build_outputs():
    """Reshape scalability_raw.csv into results_live.pickle (same schema as
    experiment_results/experiment_1_scalability/results.pickle: dataset -> algo -> {'row': (xs, ys),
    'col': (xs, ys)}) and render scalability_live.png. Out-of-memory points
    are left out, i.e. drawn as missing entries."""
    raw = _raw_df()
    if raw.empty:
        log("no data points yet, skipping output build")
        return
    raw = raw[raw["status"] != "oom"]
    combined = {}
    for name in ALL_DATASETS:
        per_algo = {}
        for algo in ALL_ALGORITHMS:
            algo_rows = raw[(raw["dataset"] == name) & (raw["algo"] == algo)]
            if algo_rows.empty:
                continue
            series = {}
            for series_type in ALL_SERIES:
                sub = algo_rows[algo_rows["series_type"] == series_type].sort_values("x")
                series[series_type] = (sub["x"].tolist(), sub["elapsed_seconds"].tolist())
            per_algo[algo] = series
        if per_algo:
            combined[name] = per_algo

    # results.pickle's own key spelling, so squareplot.py can read either file
    as_published = {
        _PICKLE_DATASET_KEYS.get(name, name): {_PICKLE_ALGO_KEYS.get(a, a): s for a, s in per_algo.items()}
        for name, per_algo in combined.items()
    }
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(as_published, f)
    log(f"wrote {PICKLE_PATH}")
    _plot(combined)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m diada.experiments.experiment_1_scalability.scalability_benchmark",
        description="Experiment 1: LIMA (DIADA) vs. Apriori/FP-Growth row/column scalability (paper Figure 4).",
    )
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS, default=list(ALL_ALGORITHMS))
    parser.add_argument("--series", nargs="+", choices=ALL_SERIES, default=list(ALL_SERIES),
                        help="row = eps^-1 sweep on the original dataset, col = column-count sweep")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timed repeats per point (default: 5)")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS,
                        help="seconds; a series stops after its first point above this (default: 1800)")
    parser.add_argument("--max-cols", type=int, default=MAX_COLS,
                        help="last column count of the col series (default: 98)")
    parser.add_argument("--row-points", type=int, default=APROX_INV_POINTS,
                        help="number of log-spaced eps^-1 points in 1e3..--row-max (default: 20)")
    parser.add_argument("--row-max", type=float, default=APROX_INV_MAX,
                        help="last eps^-1 point of the row series (default: 1e8)")
    parser.add_argument("--row-cols", type=int, default=None,
                        help="run the row series on a fixed random expansion to this many columns "
                             "instead of the original dataset (default: original dataset, as in the paper)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: results/experiment_1_scalability, or "
                             "results/experiment_1_scalability_quick with --quick)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="labelled-input cache (default: <out-dir>/cache)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: 1 repeat, 60 s limit, col series up to 10 columns, row series "
                             "3 points in 1e3..1e5 on an 8-column expansion")
    parser.add_argument("--plot-only", action="store_true",
                        help="only rebuild results_live.pickle and the figure from scalability_raw.csv")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    global DATASET_NAMES, ALGORITHMS, SERIES, REPEATS, TIME_LIMIT_SECONDS, APROX_INV_POINTS, MAX_COLS
    global ROW_APROX_INV_MAX, ROW_COLS, PAPER_AXES
    args = _parse_args(argv)
    if args.quick:
        args.repeats, args.time_limit, args.max_cols = 1, 60.0, 10
        args.row_points, args.row_max, args.row_cols = 3, 1e5, 8
    DATASET_NAMES = list(args.datasets)
    ALGORITHMS = tuple(a for a in ALL_ALGORITHMS if a in args.algorithms)
    SERIES = tuple(s for s in ALL_SERIES if s in args.series)
    REPEATS, TIME_LIMIT_SECONDS = args.repeats, args.time_limit
    APROX_INV_POINTS, MAX_COLS = args.row_points, args.max_cols
    ROW_APROX_INV_MAX, ROW_COLS = args.row_max, args.row_cols
    PAPER_AXES = MAX_COLS >= 98 and ROW_APROX_INV_MAX >= APROX_INV_MAX
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR.with_name(_DEFAULT_OUT_DIR.name + "_quick") if args.quick else _DEFAULT_OUT_DIR
    _set_out_dir(out_dir, args.cache_dir if args.cache_dir is not None else (
        Path(os.environ["CACHE_DIR"]) if "CACHE_DIR" in os.environ else None))

    if args.plot_only:
        build_outputs()
        log("ALL DONE")
        return

    log(f"Experiment 1 (scalability) -- datasets={DATASET_NAMES} algorithms={ALGORITHMS} series={SERIES} "
        f"repeats={REPEATS} time_limit={TIME_LIMIT_SECONDS:.0f}s out_dir={OUT_DIR}")
    for name in DATASET_NAMES:
        base_dataset = get_or_label_dataset(name)
        for algo in ALGORITHMS:
            if "row" in SERIES:
                run_sample_size_series(name, base_dataset, algo)
            if "col" in SERIES:
                run_column_series(name, base_dataset, algo)

    build_outputs()
    log("ALL DONE")


if __name__ == "__main__":
    main()
