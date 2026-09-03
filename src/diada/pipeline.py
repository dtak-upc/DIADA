"""
The "system": given one CSV + target column, run DIADA, build cleaned
dataset(s) (UN kept/leftover, MN per-component/leftover, or both -- UN =
univariate-noise cleaning, MN = multivariate-noise cleaning; renamed from
the old "CF1"/"CF2" naming, see core/dataset_builder.py's module
docstring), and optionally run ARDA's feature-selection pipeline
(fit_transform_full_table) on each one.

This is new -- there was previously no way to run this pipeline on anything
other than the fixed benchmarks.yaml sweep. See cli.py for the command-line
wrapper over this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .core.dataset_builder import DEFAULT_THRESHOLD, build_un_datasets, build_mn_datasets
from .core.diada_tool import invoke_diada
from .arda.arda import ARDA


@dataclass
class PipelineConfig:
    csv_path: str
    target_column: str
    task: str  # "classification" | "regression"
    metric: str = "accuracy"
    mode: str = "both"  # "un" | "mn" | "both"
    threshold: float = DEFAULT_THRESHOLD
    num_buckets: int = 10
    run_arda: bool = True
    num_features: int = 10
    output_dir: str = "diada_pipeline_output"
    engine: str = "jar"  # "jar" (default, needs Java) or "lima" (pure Python, opt-in) -- see
                         # core/diada_tool.py's module docstring: NOT numerically interchangeable
    rifs_k: int = 10
    rifs_eta: float = 0.2
    cv: int = 3
    random_state: int = 0
    verbose: bool = True


@dataclass
class PipelineResult:
    dataset_paths: Dict[str, Path]
    datasets: Dict[str, pd.DataFrame]
    arda_results: Dict[str, pd.DataFrame] = field(default_factory=dict)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full single-CSV system end to end; see module docstring."""
    if config.mode not in ("un", "mn", "both"):
        raise ValueError(f"mode must be 'un', 'mn' or 'both', got {config.mode!r}")

    base_dataset = pd.read_csv(config.csv_path)
    if config.target_column not in base_dataset.columns:
        raise ValueError(f"target_column '{config.target_column}' not found in {config.csv_path}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(f"Running DIADA on {config.csv_path} ({base_dataset.shape[0]} rows, {base_dataset.shape[1]} cols)...")
    soundness_df = invoke_diada(base_dataset, config.num_buckets, engine=config.engine, verbose=config.verbose)
    soundness_df.to_csv(output_dir / "soundness.csv", index=False)

    datasets: Dict[str, pd.DataFrame] = {}
    if config.mode in ("un", "both"):
        un = build_un_datasets(soundness_df, base_dataset, config.target_column, threshold=config.threshold)
        datasets["un_kept"] = un["kept"]
        datasets["un_leftover"] = un["leftover"]
    if config.mode in ("mn", "both"):
        mn = build_mn_datasets(soundness_df, base_dataset, config.target_column, threshold=config.threshold)
        for name, df in mn.items():
            datasets[f"mn_{name}"] = df

    dataset_paths: Dict[str, Path] = {}
    for name, df in datasets.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        dataset_paths[name] = path

    arda_results: Dict[str, pd.DataFrame] = {}
    if config.run_arda:
        for name, df in datasets.items():
            if config.target_column not in df.columns:
                if config.verbose:
                    print(f"Skipping ARDA on '{name}': no target column present.")
                continue
            if config.verbose:
                print(f"Running ARDA on '{name}' ({df.shape[1] - 1} candidate feature(s), "
                      f"requesting num_features={config.num_features})...")
            arda = ARDA(
                target_col=config.target_column, coreset_size=None,
                coreset_method="stratified" if config.task == "classification" else "uniform",
                task=config.task, metric=config.metric, rifs_k=config.rifs_k, rifs_eta=config.rifs_eta,
                cv=config.cv, random_state=config.random_state, verbose=config.verbose, benchmark=name,
            )
            result_df = arda.fit_transform_full_table(
                df, config.target_column, config.task, config.metric, num_features=config.num_features,
            )
            result_df.insert(0, "dataset", name)
            arda_results[name] = result_df

    if arda_results:
        pd.concat(arda_results.values(), ignore_index=True).to_csv(output_dir / "arda_results.csv", index=False)

    return PipelineResult(dataset_paths=dataset_paths, datasets=datasets, arda_results=arda_results)
