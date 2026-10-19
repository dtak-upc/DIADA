"""
Experiment 4 (paper Section 6.4, "One composition, any target"): one
untargeted DIADA composition, evaluated against two synthetic targets
defined afterwards.

For every dataset in benchmarks.yaml this:
  1. injects UN (univariate) noise and the two synthetic targets
     (generate_benchmark_noise_un_synth_targets, random_state=0 -- the same
     call experiment_3_diada_arda/generate_benchmarks.py makes);
  2. runs DIADA once on that table with the jar (NUM_BUCKETS=10), and keeps
     every column with a soundness relation above threshold (UN cleaning,
     build_un_datasets(include_synth_targets=True), which drops the real
     target). DIADA never knows which column is a target: the composition is
     computed once and shared by both targets;
  3. for each entry of benchmark_2targets.yaml (<dataset>_synthetic_target_<k>),
     runs ARDA's 1..max_features sweep with synthetic_target_<k> as target on
     three variants:
        <entry>_base   original dataset + both synthetic targets (no noise)
        <entry>_noise  the UN-noised table, not cleaned (pre-composition)
        <entry>_clean  the DIADA-cleaned table (post-composition)

This is the "benchmark" naming of the frozen results in
experiment_results/experiment_4_synthetic_targets/results_2targets_*.csv,
which this script never overwrites. Variant contents were reverse-engineered
from those results (candidate-feature counts and feature names) and checked
against the repo's data/ files:
  - base keeps the real target as a candidate feature and an extra
    "Unnamed: 0" column (0..n-1): the original base table was evidently
    written with its index and read back. Kept here so the candidate set
    (and hence the ranking) matches.
  - noise keeps the real target and the other synthetic target as features.
  - clean contains the other synthetic target only if DIADA kept it.
  - ARDA reads every variant back from its CSV rather than using the
    in-memory frame: the published results were computed from the CSVs, and
    the round trip's last-digit float changes are enough to reorder
    near-tied RIFS scores on regression benchmarks.
  - The data/output/output_*_2targets.csv files in the repo are from an
    older noise scheme (noise_1..noise_N columns) and were NOT used for the
    published results; data/output/output_*_un_synth_targets.csv were, and
    this script's jar output matches them (checked on bank and diamonds).

Two modes:
  - default: full sweep, one AutoGluon model per feature count (up to 40 per
    variant at up to 180 s each -- expect days for the full suite). Written
    to <out-dir>/reproduced_2targets_{classification,regression}.csv.
  - --selection-only: only the RIFS ranking, no model training. ARDA's
    selected_features for N features is exactly the top N of that ranking,
    computed before any model is trained, so this reproduces the
    selected_features column (metrics left empty) in minutes per dataset.
    Written to <out-dir>/selection_2targets_{classification,regression}.csv.

Both modes are resumable (a variant already in the output CSV is skipped)
and finish by comparing against the frozen results (--compare-only does just
that). Note: the published selected_features strings also list the target
column; the comparison strips it.

Intermediate tables (noisy, jar output, cleaned) are written to
<out-dir>/data/, not to the repo's data/ folder.

Run:            python -m diada.experiments.experiment_4_synthetic_targets.run_synthetic_targets
Features only:  ... --selection-only
Subset:         ... --datasets diamonds bank --max-features 10
Compare only:   ... --compare-only [--selection-only]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ... import paths
from ...arda.arda import ARDA
from ...core.dataset_builder import build_un_datasets
from ...core.diada_tool import invoke_diada
from ..experiment_3_diada_arda.noise_benchmarks import generate_benchmark_noise_un_synth_targets

NUM_BUCKETS = 10
MAX_FEATURES_DEFAULT = 40
SYNTH_TARGETS = ("synthetic_target_1", "synthetic_target_2")
VARIANTS = ("base", "noise", "clean")
PUBLISHED_DIR = paths.REPO_ROOT / "experiment_results" / "experiment_4_synthetic_targets"
TASKS = ("classification", "regression")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _entries_by_dataset(entries: Dict[str, dict]) -> Dict[str, List[str]]:
    """{'jannis': ['jannis_synthetic_target_1', 'jannis_synthetic_target_2'], ...}, in yaml order."""
    grouped: Dict[str, List[str]] = {}
    for entry, cfg in entries.items():
        dataset = entry[: -len("_" + cfg["target_column"])]
        grouped.setdefault(dataset, []).append(entry)
    return grouped


def build_variants(dataset: str, real_target: str, data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Noise injection + one jar DIADA run + UN cleaning for `dataset`;
    returns {"base", "noise", "clean"} DataFrames (each still holding both
    synthetic targets). Writes the intermediate tables under `data_dir`."""
    for sub in ("base_synth_targets", "noise_un_synth_targets", "output", "cleaned_un_synth_targets"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    base_dataset = pd.read_csv(paths.BASE_DATASETS_DIR / f"{dataset}.csv")
    noisy, _ = generate_benchmark_noise_un_synth_targets(base_dataset, real_target)
    noisy_path = data_dir / "noise_un_synth_targets" / f"{dataset}_un_noise_synth_targets.csv"
    noisy.to_csv(noisy_path, index=False)

    log(f"[{dataset}] running DIADA (jar) on {noisy.shape[1]} columns...")
    diada_output = invoke_diada(noisy, NUM_BUCKETS, engine="jar")
    diada_output.to_csv(data_dir / "output" / f"output_{dataset}_un_synth_targets.csv", index=False)

    clean = build_un_datasets(diada_output, noisy, real_target, include_synth_targets=True)["kept"]
    clean_path = data_dir / "cleaned_un_synth_targets" / f"{dataset}_un_synth_targets_cleaned_synth_targets.csv"
    clean.to_csv(clean_path, index=False)

    # original columns + both synthetic targets. Written WITH its index, as the
    # published base variant evidently was -- reading it back is what yields
    # its extra "Unnamed: 0" column (see module docstring).
    base = noisy[list(base_dataset.columns) + list(SYNTH_TARGETS)]
    base_path = data_dir / "base_synth_targets" / f"{dataset}_base_synth_targets.csv"
    base.to_csv(base_path, index=True)

    # ARDA runs on the tables as read back from these CSVs, not the in-memory
    # frames: the CSV round trip perturbs float values in the last digits,
    # which is enough to reorder near-tied RIFS scores. The published results
    # were computed from the CSVs (checked on avocado_sales: 40/40 feature
    # counts match from the CSVs vs. 32/40 from memory).
    return {"base": pd.read_csv(base_path), "noise": pd.read_csv(noisy_path), "clean": pd.read_csv(clean_path)}


def _make_arda(target: str, task: str, metric: str, benchmark: str, verbose: bool) -> ARDA:
    # same construction as experiment_3_diada_arda/run_paper_experiments.py
    return ARDA(target_col=target, coreset_size=None,
                coreset_method="stratified" if task == "classification" else "uniform",
                task=task, metric=metric, rifs_k=10, rifs_eta=0.2, cv=3,
                verbose=verbose, benchmark=benchmark)


def _selection_rows(df: pd.DataFrame, target: str, task: str, metric: str, benchmark: str,
                    max_features: int, verbose: bool) -> pd.DataFrame:
    """The RIFS ranking ARDA's sweep would use, expanded into one row per
    feature count -- the sweep's selected_features without training models."""
    arda = _make_arda(target, task, metric, benchmark, verbose)
    arda.target_col, arda._task_resolved = target, task
    candidates = [c for c in df.columns if c != target]
    start = time.time()
    ranking = arda.select_features(df, candidates)
    elapsed = time.time() - start
    ranked = ranking["feature"].tolist()
    counts = range(1, min(max_features, len(candidates)) + 1)
    return pd.DataFrame([
        {"benchmark": benchmark, "problem": task, "augmentation_time": elapsed,
         "num_new_features": n, "selected_features": ",".join(sorted(ranked[:n]))}
        for n in counts
    ])


def _output_path(out_dir: Path, task: str, selection_only: bool) -> Path:
    prefix = "selection" if selection_only else "reproduced"
    return out_dir / f"{prefix}_2targets_{task}.csv"


def _done_benchmarks(out_dir: Path, selection_only: bool) -> set:
    done = set()
    for task in TASKS:
        p = _output_path(out_dir, task, selection_only)
        if p.exists():
            done |= set(pd.read_csv(p, usecols=["benchmark"])["benchmark"])
    return done


def _append(rows: pd.DataFrame, out_dir: Path, task: str, selection_only: bool) -> None:
    p = _output_path(out_dir, task, selection_only)
    rows.to_csv(p, mode="a" if p.exists() else "w", header=not p.exists(), index=False)


def run_for_dataset(dataset: str, real_target: str, entries: List[str], cfgs: Dict[str, dict], out_dir: Path,
                    max_features: int, selection_only: bool, verbose: bool) -> None:
    benchmarks = [f"{e}_{v}" for e in entries for v in VARIANTS]
    done = _done_benchmarks(out_dir, selection_only)
    if all(b in done for b in benchmarks):
        log(f"[{dataset}] all variants already done -- skipping")
        return

    variants = build_variants(dataset, real_target, out_dir / "data")
    for entry in entries:
        cfg = cfgs[entry]
        target, task, metric = cfg["target_column"], cfg["task"], cfg["metric"]
        for variant in VARIANTS:
            benchmark = f"{entry}_{variant}"
            if benchmark in done:
                continue
            df = variants[variant]
            if target not in df.columns:
                log(f"[{benchmark}] skipped: {target} not in this variant")
                continue
            log(f"[{benchmark}] {df.shape[1] - 1} candidate feature(s), "
                f"{'ranking only' if selection_only else 'full sweep'}...")
            if selection_only:
                rows = _selection_rows(df, target, task, metric, benchmark, max_features, verbose)
            else:
                rows = _make_arda(target, task, metric, benchmark, verbose).fit_transform_full_table(
                    df, target, task, metric, max_features=max_features)
            _append(rows, out_dir, task, selection_only)
            log(f"[{benchmark}] done ({len(rows)} row(s))")


def compare_with_published(out_dir: Path, selection_only: bool) -> pd.DataFrame:
    """Per benchmark: how many feature counts reproduce the published
    selected_features exactly (target column stripped from the published
    strings), and, for full runs, the mean absolute difference of the main
    metric."""
    summaries = []
    for task in TASKS:
        mine_path = _output_path(out_dir, task, selection_only)
        if not mine_path.exists():
            continue
        mine = pd.read_csv(mine_path)
        pub = pd.read_csv(PUBLISHED_DIR / f"results_2targets_{task}.csv")
        metric_col = "test_accuracy" if task == "classification" else "test_rmse"
        for benchmark, m in mine.groupby("benchmark", sort=False):
            target = next(t for t in SYNTH_TARGETS if f"_{t}_" in benchmark)
            p = pub[pub["benchmark"] == benchmark].drop_duplicates("num_new_features")
            joined = m.merge(p, on="num_new_features", suffixes=("", "_pub"))
            # features in my selection but not the published one, per feature count
            diffs = [
                len(set(a.split(",")) - (set(b.split(",")) - {target}))
                for a, b in zip(joined["selected_features"], joined["selected_features_pub"])
            ]
            same = [d == 0 for d in diffs]
            mismatched = [int(n) for n, d in zip(joined["num_new_features"], diffs) if d]
            row = {"benchmark": benchmark, "task": task, "counts_mine": len(m), "counts_published": len(p),
                   "counts_compared": len(joined), "selection_matches": sum(same), "first_mismatch_at": mismatched[0] if mismatched else None,
                   # 1 = at worst one feature swapped in/out at any count (rank-order ties)
                   "max_features_differing": max(diffs, default=0),
                   "mismatched_counts": " ".join(map(str, mismatched))}
            if not selection_only and metric_col in joined and f"{metric_col}_pub" in joined:
                row[f"mean_abs_diff_{metric_col}"] = (joined[metric_col] - joined[f"{metric_col}_pub"]).abs().mean()
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    if summary.empty:
        log("nothing to compare yet")
        return summary
    # every feature count present in both runs reproduced. Sweeps of different
    # length are flagged rather than failed: a published sweep that stopped
    # early, or a --max-features slice of this one.
    summary["all_match"] = summary["selection_matches"] == summary["counts_compared"]
    summary["published_sweep_incomplete"] = summary["counts_mine"] > summary["counts_published"]
    summary["partial_run"] = summary["counts_mine"] < summary["counts_published"]
    path = out_dir / f"comparison_{'selection' if selection_only else 'reproduced'}.csv"
    summary.to_csv(path, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string(index=False))
    log(f"{int(summary['all_match'].sum())}/{len(summary)} benchmark variants reproduce the published "
        f"selected_features at every compared feature count ({int(summary['partial_run'].sum())} of them "
        f"with fewer feature counts than published); wrote {path}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m diada.experiments.experiment_4_synthetic_targets.run_synthetic_targets",
        description="Experiment 4: one untargeted DIADA composition vs. two synthetic targets (paper Section 6.4).",
    )
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Base dataset names to run (default: every dataset in benchmark_2targets.yaml).")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES_DEFAULT, dest="max_features",
                        help=f"Sweep 1..N top-ranked features per variant (default: {MAX_FEATURES_DEFAULT}).")
    parser.add_argument("--output-dir", type=Path, default=paths.RESULTS_DIR / "experiment_4_synthetic_targets",
                        dest="output_dir", help="Default: results/experiment_4_synthetic_targets.")
    parser.add_argument("--selection-only", action="store_true",
                        help="Only compute ARDA's feature rankings (selected_features), no model training.")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare an existing run's output against the published results.")
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.compare_only:
        with open(paths.BENCHMARKS_2TARGETS_YAML) as f:
            entries = yaml.safe_load(f)
        with open(paths.BENCHMARKS_YAML) as f:
            real_targets = {name: cfg["target_column"] for name, cfg in yaml.safe_load(f).items()}

        grouped = _entries_by_dataset(entries)
        names = args.datasets if args.datasets else list(grouped)
        unknown = [n for n in names if n not in grouped]
        if unknown:
            raise ValueError(f"Unknown dataset name(s): {unknown}. Known: {list(grouped)}")

        for name in names:
            run_for_dataset(name, real_targets[name], grouped[name], entries, args.output_dir,
                            args.max_features, args.selection_only, args.verbose)

    compare_with_published(args.output_dir, args.selection_only)
    log("ALL DONE")


if __name__ == "__main__":
    main()
