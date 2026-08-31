"""
Command-line entry point for the DIADA "system": one CSV in, cleaned
dataset(s) (and optionally ARDA results) out. Installed as the `diada-clean`
console script (see pyproject.toml); also runnable as `python -m diada.cli`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .pipeline import PipelineConfig, run_pipeline



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diada-clean", description="Run DIADA cleaning (and optionally ARDA) on a single CSV.")
    parser.add_argument("--csv", required=True, dest="csv_path", help="Path to the input CSV.")
    parser.add_argument("--target", required=True, dest="target_column", help="Target/label column name.")
    parser.add_argument("--task", required=True, choices=["classification", "regression"])
    parser.add_argument("--metric", default="accuracy",
                         help="Evaluation metric passed to ARDA (accuracy/f1 for classification, "
                              "root_mean_squared_error/mae for regression).")
    parser.add_argument("--mode", default="both", choices=["cf1", "cf2", "both"], help="Which cleaning strategy to run.")
    parser.add_argument("--threshold", type=float, default=4.0, help="Soundness threshold for the DIADA graph.")
    parser.add_argument("--num-buckets", type=int, default=10, dest="num_buckets",
                         help="DIADA column-typing mode: 0 = all String, 1 = typed but unbucketed, >1 = bucket numerics into this many bins.")
    parser.add_argument("--num-features", type=int, default=10, dest="num_features",
                         help="Exact number of top-ranked features to train the model with (default: 10). "
                              "Trains a single model -- if you want the full 1..N feature-count sweep used for "
                              "the paper's reproducibility experiments instead, use "
                              "`python -m diada.experiments.run_paper_experiments`.")
    parser.add_argument("--engine", default="lima", choices=["lima", "jar"],
                         help="Soundness-scoring engine: 'lima' (default, pure Python, no Java needed) or "
                              "'jar' (the original DIADA-0.8.jar, needs Java on PATH). NOT numerically "
                              "interchangeable -- see README's 'Two soundness-scoring engines' section.")
    parser.add_argument("--output-dir", default="diada_pipeline_output", dest="output_dir")
    parser.add_argument("--no-arda", action="store_false", dest="run_arda", help="Only produce cleaned datasets, skip running ARDA.")
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    return parser


def _print_summary(result) -> None:
    print("\nGenerated datasets:")
    for name, path in result.dataset_paths.items():
        print(f"  {name}: {path}")

    if not result.arda_results:
        return

    print("\nARDA results (one model per dataset, trained with the requested num_features):")
    for name, df in result.arda_results.items():
        if df is None or df.empty:
            continue
        metric_col = next((c for c in df.columns if c.startswith("test_")), None)
        if metric_col is None:
            continue
        row = df.iloc[0]
        print(f"  {name}: {metric_col}={row[metric_col]:.4f} (num_new_features={int(row['num_new_features'])})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = PipelineConfig(
        csv_path=args.csv_path, target_column=args.target_column, task=args.task, metric=args.metric,
        mode=args.mode, threshold=args.threshold, num_buckets=args.num_buckets, engine=args.engine,
        num_features=args.num_features, output_dir=args.output_dir, run_arda=args.run_arda, verbose=args.verbose,
    )
    result = run_pipeline(config)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
