"""
Experiment 2: DIADA vs. Unsupervised Feature Selection (UFS) baselines.

Live, re-runnable port of
experiment_results/experiment_2_ufs/ufs_extended.ipynb -- the notebook that
produced experiment_results/experiment_2_ufs/ufs_results_final.csv, the
paper's noise-filtering comparison table (does DIADA drop injected noise
columns better than generic UFS methods?). See EXPERIMENTS.md for the
running list of numbered reproducibility experiments this belongs to; see
ufs_baselines.py for the ported UFS scoring functions themselves.

Differences from the notebook, all disclosed here rather than silently
applied
-----------------------------------------------------------------------
  1. Hardcoded absolute path fixed. The notebook read noise-injected input
     from `C:/Projects/arclo/data/noise_cf1/...` -- "arclo" appears to be
     this project's name before it became DIADA, and "cf1" the old name for
     what this package now calls "un" (univariate-noise cleaning -- see
     core/dataset_builder.py's module docstring for that rename). The
     identical files already exist at `data/noise_un/` in *this* repo
     (verified: all 17 names match benchmarks.yaml). This script reads from
     there via `diada.paths`, cwd-independent like the rest of the package.
  2. Full benchmark loop restored. The notebook's benchmark-running cell
     was hardcoded to `["students", "pendigits"]` (the full loop over every
     benchmarks.yaml entry was commented out), even though the committed
     ufs_results_final.csv covers all 17 -- i.e. the checked-in notebook
     state does not reproduce its own committed output. This script loops
     every entry in benchmarks.yaml.
  3. The "lima" method is real code now, and renamed "diada". In the
     frozen CSV, a method called "lima" has metrics that exactly match
     data/cleaned_un/{name}_un_cleaned.csv's column set (verified
     byte-for-byte on the `bank` benchmark) -- i.e. DIADA's own UN output,
     scored with the same noise-counting logic as the UFS methods -- but no
     code in the repo actually produces that comparison. This script adds
     it for real: invoke_diada(engine="jar", num_buckets=10) +
     build_un_datasets(threshold=4) (same defaults
     experiments/generate_benchmarks.py uses), timed the same way as every
     UFS method, scored with ufs_baselines.is_noise/count_noise. Per
     explicit instruction: LIMA is the internal algorithm name, DIADA is
     the system -- every results-facing label here uses "diada"/"DIADA",
     never "lima" (see DISPLAY_NAME_OVERRIDES below, same pattern
     experiment_1_scalability/squareplot.py already used for this exact
     purpose).
  4. cae re-enabled. The notebook's `methods` dict has cae commented out
     (`# if TORCH_AVAILABLE: methods["cae"] = ...`), even though
     ufs_results_final.csv has real cae rows -- another case of the
     checked-in notebook state not matching its own committed output. Per
     explicit instruction this script re-enables it (ufs_baselines.py:
     auto-skipped if torch isn't installed, same detect-and-skip pattern
     Experiment 1 uses for Apriori/FP-Growth).
  5. Known failures reproduced as-is, not fixed, per explicit instruction:
       - spec/mcfs/ndfs/rsr do a dense N x N eigendecomposition (N = row
         count of the *dataset*, not column count) -- this OOMs on the
         larger benchmarks (jannis, miniboone, bank, drive_diagnosis,
         covertype, default_payment, nasa in the frozen run) and is a real
         scalability ceiling of this implementation, not a bug.
       - Separately, every W-consuming method (laplacian_score, spec,
         mcfs, ndfs, rsr, agufs) failed on `diamonds` specifically in the
         frozen run with `'NoneType' object has no attribute 'sum'`,
         while every non-W method succeeded on the same dataset -- implying
         W was None there in whatever code actually produced that run.
         Root cause not identified; left to reproduce and fail the same
         way rather than guessing at a fix.
     Both are recorded in the `error` column exactly like the frozen CSV.
  6. Resumable: every (benchmark, method) row is checkpointed to
     ufs_raw.csv before moving to the next, so a long sweep survives being
     interrupted and restarted -- same pattern as
     scalability_benchmark.py.

Run with: python3 -m diada.experiments.experiment_2_ufs.ufs_benchmark
Env vars (optional): OUT_DIR (default results/experiment_2_ufs under the
repo root).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from ... import paths
from ...core.dataset_builder import DEFAULT_THRESHOLD, build_un_datasets
from ...core.diada_tool import invoke_diada
from . import ufs_baselines as ufs

GRAPH_K = 5
GRAPH_SIGMA = 1.0
MCFS_CLUSTERS = 5
UDFS_ALPHA = 0.01
N_KEEP = None  # auto elbow-cut, matching the notebook's defaults

DIADA_NUM_BUCKETS = 10  # matches experiments/generate_benchmarks.py's NUM_BUCKETS
DIADA_ENGINE = "jar"  # matches invoke_diada's own default and generate_benchmarks.py

DISPLAY_NAME_OVERRIDES = {"diada": "DIADA"}

OUT_DIR = Path(os.environ.get("OUT_DIR", str(paths.RESULTS_DIR / "experiment_2_ufs")))
RAW_CSV = OUT_DIR / "ufs_raw.csv"
SUMMARY_CSV = OUT_DIR / "ufs_summary.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Copied from experiment_results/experiment_2_ufs/ufs_extended.ipynb's analysis cell -- raw column counts
# (excluding injected noise/duplicate columns) per benchmark, used to
# compute what fraction of a dataset's real features each method kept.
ORIGINAL_FEATURE_COUNTS = {
    "bank": 16,
    "default_payment": 24,
    "jannis": 54,
    "miniboone": 50,
    "appliances": 28,
    "avocado_sales": 13,
    "diamonds": 9,
    "house_sales": 17,
    "nasa": 22,
    "pol": 48,
    "superconductivity": 82,
    "covertype": 13,
    "drive_diagnosis": 48,
    "dry_beans": 17,
    "mice_protein": 81,
    "pendigits": 17,
    "students": 36,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame(columns=[
            "benchmark", "method", "selected_columns", "includes_target",
            "n_selected", "n_noise", "noise_ratio", "execution_time_sec", "error",
        ])
    return pd.read_csv(RAW_CSV)


def already_done(benchmark: str, method: str) -> bool:
    done = _raw_df()
    return bool(((done["benchmark"] == benchmark) & (done["method"] == method)).any())


def append_row(row: dict):
    df = pd.DataFrame([row])
    header = not RAW_CSV.exists()
    df.to_csv(RAW_CSV, mode="a" if not header else "w", header=header, index=False)


def _row_from_selected(benchmark, method, selected, target_column, elapsed, error=None):
    return {
        "benchmark": benchmark,
        "method": method,
        "selected_columns": selected,
        "includes_target": target_column in selected,
        "n_selected": len(selected),
        "n_noise": ufs.count_noise(selected),
        "noise_ratio": (ufs.count_noise(selected) / len(selected)) if selected else 0,
        "execution_time_sec": round(elapsed, 4),
        "error": error,
    }


def run_ufs_methods(benchmark: str, df: pd.DataFrame, target_column: str):
    to_run = [m for m in _method_names() if not already_done(benchmark, m)]
    if not to_run:
        return

    log(f"[{benchmark}] preparing dataframe + similarity graph...")
    X, groups = ufs.prepare_dataframe(df)
    W = ufs.build_similarity_graph(X, k=GRAPH_K, sigma=GRAPH_SIGMA)
    methods = ufs.build_methods(X, W, n_keep=N_KEEP, mcfs_clusters=MCFS_CLUSTERS, udfs_alpha=UDFS_ALPHA)

    for name, (fn, larger_better) in methods.items():
        if already_done(benchmark, name):
            continue
        log(f"[{benchmark} | {name}] running...")
        start = time.perf_counter()
        try:
            scores, selected_x = fn()
            elapsed = time.perf_counter() - start
            selected = ufs.x_indices_to_columns(selected_x, groups)
            row = _row_from_selected(benchmark, name, selected, target_column, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            row = _row_from_selected(benchmark, name, [], target_column, elapsed, error=str(e))
            log(f"[{benchmark} | {name}] ERROR: {e}")
        append_row(row)
        log(f"[{benchmark} | {name}] done in {row['execution_time_sec']:.3f}s, "
            f"n_selected={row['n_selected']} n_noise={row['n_noise']}")


def run_diada(benchmark: str, df: pd.DataFrame, target_column: str):
    if already_done(benchmark, "diada"):
        return
    log(f"[{benchmark} | diada] running (engine={DIADA_ENGINE}, num_buckets={DIADA_NUM_BUCKETS})...")
    start = time.perf_counter()
    try:
        soundness = invoke_diada(df, DIADA_NUM_BUCKETS, engine=DIADA_ENGINE, verbose=False)
        un = build_un_datasets(soundness, df, target_column, threshold=DEFAULT_THRESHOLD)
        elapsed = time.perf_counter() - start
        selected = un["kept"].columns.tolist()
        row = _row_from_selected(benchmark, "diada", selected, target_column, elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - start
        row = _row_from_selected(benchmark, "diada", [], target_column, elapsed, error=str(e))
        log(f"[{benchmark} | diada] ERROR: {e}")
    append_row(row)
    log(f"[{benchmark} | diada] done in {row['execution_time_sec']:.3f}s, "
        f"n_selected={row['n_selected']} n_noise={row['n_noise']}")


def _method_names():
    dummy_methods = ufs.build_methods(np.zeros((6, 1)), np.zeros((6, 6)))
    return list(dummy_methods.keys())


def _confidence_interval(series: pd.Series, confidence: float = 0.95):
    series = series.dropna()
    n = len(series)
    if n < 2:
        return (np.nan, np.nan)
    mean = series.mean()
    sem = stats.sem(series)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (mean - margin, mean + margin)


def build_summary():
    raw = _raw_df()
    if raw.empty:
        log("no data points yet, skipping summary")
        return

    df = raw.drop(columns=["selected_columns"]).copy()
    df["includes_target"] = df["includes_target"].astype(str).map({"True": True, "False": False}).astype(float)

    df["n_true_selected"] = df["n_selected"] - df["n_noise"]
    df["selected_ratio"] = df.apply(
        lambda row: row["n_true_selected"] / ORIGINAL_FEATURE_COUNTS[row["benchmark"]]
        if row["benchmark"] in ORIGINAL_FEATURE_COUNTS else np.nan,
        axis=1,
    )

    summary = df.groupby("method").agg({
        "n_selected": "mean",
        "n_noise": "mean",
        "noise_ratio": "mean",
        "execution_time_sec": "mean",
        "selected_ratio": ["mean", "min"],
        "includes_target": lambda x: x.mean() * 100,
    }).reset_index()

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    ci_selected = (
        df.groupby("method")["selected_ratio"].apply(_confidence_interval).reset_index(name="ci")
    )
    ci_selected["selected_ratio_ci_low"] = ci_selected["ci"].str[0]
    ci_selected["selected_ratio_ci_high"] = ci_selected["ci"].str[1]
    ci_selected = ci_selected.drop(columns="ci")

    ci_noise = (
        df.groupby("method")["noise_ratio"].apply(_confidence_interval).reset_index(name="ci")
    )
    ci_noise["noise_ratio_ci_low"] = ci_noise["ci"].str[0]
    ci_noise["noise_ratio_ci_high"] = ci_noise["ci"].str[1]
    ci_noise = ci_noise.drop(columns="ci")

    summary = summary.merge(ci_selected, on="method", how="left").merge(ci_noise, on="method", how="left")

    summary = summary.rename(columns={
        "execution_time_sec_mean": "execution_time",
        "n_selected_mean": "n_selected",
        "n_noise_mean": "n_noise",
        "noise_ratio_mean": "noise_ratio",
        "selected_ratio_mean": "selected_ratio_mean",
        "selected_ratio_min": "selected_ratio_min",
        "includes_target_<lambda>": "includes_target_pct",
    })

    summary["selected_ratio_ci"] = summary["selected_ratio_ci_high"] - summary["selected_ratio_mean"]
    summary["noise_ratio_ci"] = summary["noise_ratio_ci_high"] - summary["noise_ratio"]

    summary["selected_ratio"] = summary.apply(
        lambda row: f"{row['selected_ratio_mean']:.3f} (± {row['selected_ratio_ci']:.3f})", axis=1)
    summary["noise_ratio"] = summary.apply(
        lambda row: f"{row['noise_ratio']:.3f} (± {row['noise_ratio_ci']:.3f})", axis=1)

    summary = summary.drop(columns=[
        "selected_ratio_mean", "selected_ratio_ci_low", "selected_ratio_ci_high", "selected_ratio_ci",
        "noise_ratio_ci_low", "noise_ratio_ci_high", "noise_ratio_ci",
    ])

    summary["method"] = summary["method"].map(lambda m: DISPLAY_NAME_OVERRIDES.get(m, m))

    summary = summary.round(3).sort_values(by=["noise_ratio", "n_selected"], ascending=[True, False])

    summary.to_csv(SUMMARY_CSV, index=False)
    log(f"wrote {SUMMARY_CSV}")
    print(summary.to_string(index=False))


def main():
    with open(paths.BENCHMARKS_YAML) as f:
        benchmarks = yaml.safe_load(f)

    log(f"Experiment 2 (UFS comparison) -- {len(benchmarks)} benchmarks, "
        f"{len(_method_names())} UFS methods + diada")
    if ufs.cae_score is None:
        log("  cae: NOT AVAILABLE (torch not installed -- skipping)")

    for name in sorted(benchmarks.keys()):
        target_column = benchmarks[name]["target_column"]
        noise_csv = paths.DATA_DIR / "noise_un" / f"{name}_un_noise.csv"
        if not noise_csv.exists():
            log(f"[{name}] SKIPPED -- {noise_csv} not found (download data/, see README)")
            continue

        df = pd.read_csv(noise_csv)
        run_ufs_methods(name, df, target_column)
        run_diada(name, df, target_column)

    build_summary()
    log("ALL DONE")


if __name__ == "__main__":
    main()
