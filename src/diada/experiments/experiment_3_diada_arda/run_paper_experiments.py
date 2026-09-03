"""
Reproduce the exact experiment sweep the paper's results were built from.

For every dataset in data/base_datasets (config per benchmarks.yaml), this
runs ARDA's full feature-count sweep (1..max_features) on five variants of
that dataset:

    <name>            raw dataset, no injected noise, no DIADA cleaning (baseline)
    <name>_un_dirty   univariate noise (UN) injected, NOT cleaned
    <name>_un_clean   UN injected, then DIADA-cleaned (kept columns only)
    <name>_mn_dirty   multivariate/spurious-cluster noise (MN) injected, NOT cleaned
    <name>_mn_clean   MN injected, then DIADA-cleaned (target's own cluster only)

This is exactly the "benchmark" naming convention already used in
experiment_results/experiment_3_diada_arda/results_final_regression.csv /
results_final_classification.csv (e.g. "house_sales_un_clean"), and it
reuses the same noise-injection / DIADA-cleaning code as
experiment_3_diada_arda/generate_benchmarks.py -- specifically it calls
generate_for_benchmark() from that module so the noisy/cleaned CSVs written
under data/ come out identical to running that script directly, and noise
generation isn't duplicated or allowed to drift (both use the same
random_state=0 default).

IMPORTANT: this does NOT overwrite the frozen
experiment_results/experiment_3_diada_arda/results_final_regression.csv /
results_final_classification.csv -- those are the paper's original
reference results. It writes results/reproduced_regression.csv and
results/reproduced_classification.csv instead (configurable via
--output-dir), so a fresh run can always be diffed against the originals.

This is by far the most expensive entry point in this package: the default
call below trains up to 40 AutoGluon models per variant, 5 variants per
dataset, for every dataset in benchmarks.yaml (17 by default) -- expect it to
take a long time. Use --datasets / --max-features to run a smaller slice
(e.g. while testing).

Run the full sweep with:   python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments
Run a quick subset with:   python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments --datasets diamonds --max-features 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from ... import paths
from ...arda.arda import ARDA
from .generate_benchmarks import generate_for_benchmark

MAX_FEATURES_DEFAULT = 40


def _run_arda(df: pd.DataFrame, target_column: str, task: str, metric: str, benchmark_name: str,
              max_features: int, rifs_k: int = 10, rifs_eta: float = 0.2, cv: int = 3, verbose: bool = True) -> pd.DataFrame:
    """One ARDA sweep (1..max_features) on `df`, tagged with `benchmark_name` --
    same construction arda/run_benchmarks.py uses for the plain, un-noised sweep."""
    arda = ARDA(
        target_col=target_column, coreset_size=None,
        coreset_method="stratified" if task == "classification" else "uniform",
        task=task, metric=metric, rifs_k=rifs_k, rifs_eta=rifs_eta, cv=cv,
        verbose=verbose, benchmark=benchmark_name,
    )
    return arda.fit_transform_full_table(df, target_column, task, metric, max_features=max_features)


def run_for_dataset(benchmark_name: str, target_column: str, task: str, metric: str,
                     max_features: int = MAX_FEATURES_DEFAULT, verbose: bool = True) -> pd.DataFrame:
    """Run all 5 variants for one dataset and return the concatenated results
    (same columns as results_final_*.csv: benchmark, problem, ..., num_new_features,
    selected_features)."""
    # generate_for_benchmark() does the noise injection + DIADA cleaning (writing
    # the same data/noise_un/, data/cleaned_un/, etc. artifacts
    # experiments/generate_benchmarks.py always has), and hands back the
    # DataFrames we need so we don't redo that work.
    if verbose:
        print(f"=== {benchmark_name}: generating noisy + cleaned variants ===")
    built = generate_for_benchmark(benchmark_name, target_column)

    variants = [
        (benchmark_name, built["base_dataset"]),
        (f"{benchmark_name}_un_dirty", built["noise_un"]),
        (f"{benchmark_name}_un_clean", built["un_kept"]),
        (f"{benchmark_name}_mn_dirty", built["noise_mn"]),
        (f"{benchmark_name}_mn_clean", built["mn_datasets"]["target_cluster"]),
    ]

    results = []
    for variant_name, df in variants:
        if target_column not in df.columns:
            if verbose:
                print(f"Skipping '{variant_name}': target column '{target_column}' not present.")
            continue
        if verbose:
            print(f"--- ARDA sweep: {variant_name} ({df.shape[1] - 1} candidate feature(s), "
                  f"max_features={max_features}) ---")
        results.append(_run_arda(df, target_column, task, metric, variant_name, max_features, verbose=verbose))

    return pd.concat(results, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments",
        description="Reproduce the paper's full noise/clean/ARDA experiment sweep across data/base_datasets.",
    )
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Subset of benchmark names to run (default: every dataset in benchmarks.yaml).")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES_DEFAULT, dest="max_features",
                         help=f"Sweep 1..N top-ranked features per variant (default: {MAX_FEATURES_DEFAULT}, "
                              "same as the original paper runs).")
    parser.add_argument("--output-dir", default=str(paths.RESULTS_DIR), dest="output_dir",
                         help="Where to write reproduced_regression.csv / reproduced_classification.csv "
                              "(default: results/). Never overwrites results_final_*.csv.")
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    args = parser.parse_args(argv)

    with open(paths.BENCHMARKS_YAML) as f:
        benchmarks_data = yaml.safe_load(f)

    names = args.datasets if args.datasets else list(benchmarks_data.keys())
    unknown = [n for n in names if n not in benchmarks_data]
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {unknown}. Known benchmarks: {list(benchmarks_data.keys())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name in names:
        cfg = benchmarks_data[name]
        df_result = run_for_dataset(
            name, cfg["target_column"], cfg["task"], cfg["metric"],
            max_features=args.max_features, verbose=args.verbose,
        )
        all_results.append(df_result)

    combined = pd.concat(all_results, ignore_index=True)

    for task_name, filename in (("classification", "reproduced_classification.csv"), ("regression", "reproduced_regression.csv")):
        subset = combined[combined["problem"] == task_name]
        if subset.empty:
            continue
        out_path = output_dir / filename
        subset.to_csv(out_path, index=False)
        print(f"Wrote {len(subset)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
