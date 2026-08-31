"""
Sweep every benchmark in benchmarks.yaml through noise injection -> DIADA ->
cleaning, writing the exact same files this project has always produced
under data/ (reproducibility path). Was benchmarks_generation/main.py.

The only behavioral additions: every cleaning step now also writes the new
"leftover"/per-component datasets (build_cf1_datasets / build_cf2_datasets in
core.dataset_builder) alongside the existing outputs, to new sibling folders
that don't touch any pre-existing path or filename:

    data/cleaned_cf1_leftover/                  (new)
    data/cleaned_cf1_synth_targets_leftover/     (new)
    data/cleaned_cf2_components/                 (new: every component, target's own included)

generate_for_benchmark() also now returns the in-memory DataFrames it built
(base/noisy/cleaned), so experiments.run_paper_experiments can reuse them
directly instead of re-running noise injection + DIADA cleaning itself.

Run with:  python -m diada.experiments.generate_benchmarks
"""

from typing import Dict

import yaml
import pandas as pd

from .. import paths
from ..core.diada_tool import invoke_diada
from ..core.dataset_builder import build_cf1_datasets, build_cf2_datasets, visualize_clusters
from .noise_benchmarks import (
    generate_benchmark_noise_cf1,
    generate_benchmark_noise_cf1_synth_targets,
    generate_benchmark_noise_cf2,
)

NUM_BUCKETS = 10

OUTPUT_DIRS = [
    paths.DATA_DIR / "output",
    paths.DATA_DIR / "noise_cf1", paths.DATA_DIR / "cleaned_cf1", paths.DATA_DIR / "cleaned_cf1_leftover",
    paths.DATA_DIR / "noise_cf1_synth_targets", paths.DATA_DIR / "cleaned_cf1_synth_targets",
    paths.DATA_DIR / "cleaned_cf1_synth_targets_leftover",
    paths.DATA_DIR / "noise_cf2", paths.DATA_DIR / "cleaned_cf2", paths.DATA_DIR / "cleaned_cf2_components",
    paths.DATA_DIR / "clusters",
]


def _run_diada_and_save(df: pd.DataFrame, file_name_diada: str) -> pd.DataFrame:
    """invoke_diada + write the soundness table to the same data/output/output_*.csv path
    the original script always wrote, then return it for immediate use.

    Pinned to engine="jar" (rather than following invoke_diada's new "lima"
    default) so this reproducibility path keeps using the exact same engine
    that produced results/results_final_*.csv -- LIMA is a randomized
    approximation that was empirically found to select a different (though
    heavily overlapping) set of "sound" column pairs than the jar (see
    core/diada_tool.py's module docstring), so silently switching engines
    here would break exact continuity with the paper's published numbers.
    Pass engine="lima" explicitly if you want this sweep on the new engine
    instead."""
    diada_output = invoke_diada(df, NUM_BUCKETS, engine="jar")
    diada_output.to_csv(paths.OUTPUT_DIR / f"output_{file_name_diada}.csv", index=False)
    return diada_output


def generate_for_benchmark(benchmark_name: str, target_column: str) -> Dict[str, object]:
    """Generate + write every noisy/cleaned artifact for one benchmark (unchanged
    behavior), and additionally RETURN the key in-memory DataFrames so callers that
    need them -- e.g. experiments.run_paper_experiments -- don't have to re-run
    noise injection / DIADA cleaning themselves (which would burn time and, being
    randomized, could in principle drift from what got written to disk here)."""
    print(benchmark_name)
    file_name = benchmark_name

    benchmark_path = paths.BASE_DATASETS_DIR / f"{file_name}.csv"
    base_dataset = pd.read_csv(benchmark_path)

    # ---- Noisy CF1 ----
    print("Generating UN benchmark")
    noise_cf1, report_cf1 = generate_benchmark_noise_cf1(base_dataset, target_column)
    noise_cf1.to_csv(paths.DATA_DIR / "noise_cf1" / f"{file_name}_cf1_noise.csv", index=False)

    # ---- Cleaned CF1 (+ leftover) ----
    print("Cleaning UN benchmark")
    file_name_diada = f"{file_name}_cf1"
    diada_output = _run_diada_and_save(noise_cf1, file_name_diada)

    cf1_datasets = build_cf1_datasets(diada_output, noise_cf1, target_column)
    cf1_datasets["kept"].to_csv(paths.DATA_DIR / "cleaned_cf1" / f"{file_name_diada}_cleaned.csv", index=False)
    cf1_datasets["leftover"].to_csv(paths.DATA_DIR / "cleaned_cf1_leftover" / f"{file_name_diada}_leftover.csv", index=False)

    # ---- CF1 with synthetic targets (noisy and cleaned + leftover) ----
    print("Generating UN benchmark with synthetic targets")
    noise_cf1_synth_targets, _ = generate_benchmark_noise_cf1_synth_targets(base_dataset, target_column)
    noise_cf1_synth_targets.to_csv(paths.DATA_DIR / "noise_cf1_synth_targets" / f"{file_name}_cf1_noise_synth_targets.csv", index=False)

    file_name_diada = f"{file_name}_cf1_synth_targets"
    diada_output = _run_diada_and_save(noise_cf1_synth_targets, file_name_diada)

    print("Cleaning UN benchmark with synthetic targets")
    cf1_synth_datasets = build_cf1_datasets(diada_output, noise_cf1_synth_targets, target_column, include_synth_targets=True)
    cf1_synth_datasets["kept"].to_csv(paths.DATA_DIR / "cleaned_cf1_synth_targets" / f"{file_name_diada}_cleaned_synth_targets.csv", index=False)
    cf1_synth_datasets["leftover"].to_csv(paths.DATA_DIR / "cleaned_cf1_synth_targets_leftover" / f"{file_name_diada}_leftover_synth_targets.csv", index=False)

    # ---- Noisy CF2 = CF1 + spurious correlated cluster ----
    print("Generating MN benchmark")
    noise_cf2, report_cf2 = generate_benchmark_noise_cf2(noise_cf1, report_cf1)
    noise_cf2.to_csv(paths.DATA_DIR / "noise_cf2" / f"{file_name}_cf2_noise.csv", index=False)

    # ---- Cleaned CF2 (target's cluster + every other cluster + leftover) ----
    print("Cleaning MN benchmark")
    file_name_diada = f"{file_name}_cf2"
    diada_output = _run_diada_and_save(noise_cf2, file_name_diada)

    cf2_datasets = build_cf2_datasets(diada_output, noise_cf2, target_column)
    # Preserve the original single-file output exactly (target's own cluster)
    cf2_datasets["target_cluster"].to_csv(paths.DATA_DIR / "cleaned_cf2" / f"{file_name_diada}_cleaned.csv", index=False)
    # New: every component (including target's, for a complete self-describing set)
    for component_name, dataset in cf2_datasets.items():
        dataset.to_csv(paths.DATA_DIR / "cleaned_cf2_components" / f"{file_name_diada}_{component_name}.csv", index=False)

    visualize_clusters(
        diada_output, target_column,
        output_path=str(paths.DATA_DIR / "clusters" / f"{benchmark_name}_cluster_map.png"),
        show=False,
    )

    return {
        "base_dataset": base_dataset,
        "noise_cf1": noise_cf1,
        "cf1_kept": cf1_datasets["kept"],
        "cf1_leftover": cf1_datasets["leftover"],
        "noise_cf1_synth_targets": noise_cf1_synth_targets,
        "cf1_synth_targets_kept": cf1_synth_datasets["kept"],
        "cf1_synth_targets_leftover": cf1_synth_datasets["leftover"],
        "noise_cf2": noise_cf2,
        "cf2_datasets": cf2_datasets,  # {"target_cluster": df, "cluster_2": df, ..., "leftover": df}
    }


def main() -> None:
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    with open(paths.BENCHMARKS_YAML) as f:
        benchmarks_data = yaml.safe_load(f)
        names = list(benchmarks_data.keys())

    for benchmark_name in ["diamonds"]:
    # for benchmark_name in names:
        target_column = benchmarks_data[benchmark_name]["target_column"]
        generate_for_benchmark(benchmark_name, target_column)


if __name__ == "__main__":
    main()
