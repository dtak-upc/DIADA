"""
Sweep every benchmark in benchmarks.yaml through noise injection -> DIADA ->
cleaning, writing the exact same files this project has always produced
under data/ (reproducibility path). Was benchmarks_generation/main.py.

Naming: "CF1"/"CF2" (old, confusing jargon) renamed throughout this package
to "UN"/"MN" -- univariate-noise / multivariate-noise cleaning, matching
what each strategy actually does (see core/dataset_builder.py's module
docstring). This renamed the on-disk data/ folder and file names too
(noise_cf1/ -> noise_un/, etc.) -- data_original/ was renamed the same way,
so the two stay in sync.

The only behavioral additions vs. the original script: every cleaning step
now also writes the new "leftover"/per-component datasets
(build_un_datasets / build_mn_datasets in core.dataset_builder) alongside
the existing outputs, to new sibling folders that don't touch any
pre-existing path or filename:

    data/cleaned_un_leftover/                  (new)
    data/cleaned_un_synth_targets_leftover/     (new)
    data/cleaned_mn_components/                 (new: every component, target's own included)

generate_for_benchmark() also now returns the in-memory DataFrames it built
(base/noisy/cleaned), so experiments.run_paper_experiments can reuse them
directly instead of re-running noise injection + DIADA cleaning itself.

Run with:  python -m diada.experiments.experiment_3_diada_arda.generate_benchmarks
"""

from typing import Dict

import yaml
import pandas as pd

from ... import paths
from ...core.diada_tool import invoke_diada
from ...core.dataset_builder import build_un_datasets, build_mn_datasets, visualize_clusters
from .noise_benchmarks import (
    generate_benchmark_noise_un,
    generate_benchmark_noise_un_synth_targets,
    generate_benchmark_noise_mn,
)

NUM_BUCKETS = 10

OUTPUT_DIRS = [
    paths.DATA_DIR / "output",
    paths.DATA_DIR / "noise_un", paths.DATA_DIR / "cleaned_un", paths.DATA_DIR / "cleaned_un_leftover",
    paths.DATA_DIR / "noise_un_synth_targets", paths.DATA_DIR / "cleaned_un_synth_targets",
    paths.DATA_DIR / "cleaned_un_synth_targets_leftover",
    paths.DATA_DIR / "noise_mn", paths.DATA_DIR / "cleaned_mn", paths.DATA_DIR / "cleaned_mn_components",
    paths.DATA_DIR / "clusters",
]


def _run_diada_and_save(df: pd.DataFrame, file_name_diada: str) -> pd.DataFrame:
    """invoke_diada + write the soundness table to the same data/output/output_*.csv path
    the original script always wrote, then return it for immediate use.

    Explicitly pinned to engine="jar" (this happens to match invoke_diada's
    own default too, but is spelled out here rather than relied on, so this
    reproducibility path keeps using the exact same engine that produced
    experiment_results/experiment_3_diada_arda/results_final_*.csv even if
    invoke_diada's default is ever changed again) -- LIMA is a randomized
    approximation that was
    empirically found to select a different (though heavily overlapping)
    set of "sound" column pairs than the jar (see core/diada_tool.py's
    module docstring), so switching engines here would break exact
    continuity with the paper's published numbers. Pass engine="lima"
    explicitly if you want this sweep on that engine instead."""
    diada_output = invoke_diada(df, NUM_BUCKETS, engine="jar")
    diada_output.to_csv(paths.OUTPUT_DIR / f"output_{file_name_diada}.csv", index=False)
    return diada_output


def generate_for_benchmark(benchmark_name: str, target_column: str) -> Dict[str, object]:
    """Generate + write every noisy/cleaned artifact for one benchmark (unchanged
    behavior), and additionally RETURN the key in-memory DataFrames so callers that
    need them -- e.g. experiments.run_paper_experiments -- don't have to re-run
    noise injection / DIADA cleaning themselves (which would burn time and, being
    randomized, could in principle drift from what got written to disk here).

    Ensures OUTPUT_DIRS exist itself (used to be the caller's job, done only in
    this module's own main() below) -- found by smoke-testing
    run_paper_experiments.py against a freshly-copied data_original/: it calls
    this function directly without ever creating these directories first, so it
    crashed with "Cannot save file into a non-existent directory" the moment
    data/ didn't already happen to have cleaned_un_leftover/ etc. from a prior
    generate_benchmarks.py run. Self-contained now regardless of caller."""
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    print(benchmark_name)
    file_name = benchmark_name

    benchmark_path = paths.BASE_DATASETS_DIR / f"{file_name}.csv"
    base_dataset = pd.read_csv(benchmark_path)

    # ---- Noisy UN (univariate noise) ----
    print("Generating UN benchmark")
    noise_un, report_un = generate_benchmark_noise_un(base_dataset, target_column)
    noise_un.to_csv(paths.DATA_DIR / "noise_un" / f"{file_name}_un_noise.csv", index=False)

    # ---- Cleaned UN (+ leftover) ----
    print("Cleaning UN benchmark")
    file_name_diada = f"{file_name}_un"
    diada_output = _run_diada_and_save(noise_un, file_name_diada)

    un_datasets = build_un_datasets(diada_output, noise_un, target_column)
    un_datasets["kept"].to_csv(paths.DATA_DIR / "cleaned_un" / f"{file_name_diada}_cleaned.csv", index=False)
    un_datasets["leftover"].to_csv(paths.DATA_DIR / "cleaned_un_leftover" / f"{file_name_diada}_leftover.csv", index=False)

    # ---- UN with synthetic targets (noisy and cleaned + leftover) ----
    print("Generating UN benchmark with synthetic targets")
    noise_un_synth_targets, _ = generate_benchmark_noise_un_synth_targets(base_dataset, target_column)
    noise_un_synth_targets.to_csv(paths.DATA_DIR / "noise_un_synth_targets" / f"{file_name}_un_noise_synth_targets.csv", index=False)

    file_name_diada = f"{file_name}_un_synth_targets"
    diada_output = _run_diada_and_save(noise_un_synth_targets, file_name_diada)

    print("Cleaning UN benchmark with synthetic targets")
    un_synth_datasets = build_un_datasets(diada_output, noise_un_synth_targets, target_column, include_synth_targets=True)
    un_synth_datasets["kept"].to_csv(paths.DATA_DIR / "cleaned_un_synth_targets" / f"{file_name_diada}_cleaned_synth_targets.csv", index=False)
    un_synth_datasets["leftover"].to_csv(paths.DATA_DIR / "cleaned_un_synth_targets_leftover" / f"{file_name_diada}_leftover_synth_targets.csv", index=False)

    # ---- Noisy MN = UN + spurious correlated cluster ----
    print("Generating MN benchmark")
    noise_mn, report_mn = generate_benchmark_noise_mn(noise_un, report_un)
    noise_mn.to_csv(paths.DATA_DIR / "noise_mn" / f"{file_name}_mn_noise.csv", index=False)

    # ---- Cleaned MN (target's cluster + every other cluster + leftover) ----
    print("Cleaning MN benchmark")
    file_name_diada = f"{file_name}_mn"
    diada_output = _run_diada_and_save(noise_mn, file_name_diada)

    mn_datasets = build_mn_datasets(diada_output, noise_mn, target_column)
    # Preserve the original single-file output exactly (target's own cluster)
    mn_datasets["target_cluster"].to_csv(paths.DATA_DIR / "cleaned_mn" / f"{file_name_diada}_cleaned.csv", index=False)
    # New: every component (including target's, for a complete self-describing set)
    for component_name, dataset in mn_datasets.items():
        dataset.to_csv(paths.DATA_DIR / "cleaned_mn_components" / f"{file_name_diada}_{component_name}.csv", index=False)

    visualize_clusters(
        diada_output, target_column,
        output_path=str(paths.DATA_DIR / "clusters" / f"{benchmark_name}_cluster_map.png"),
        show=False,
    )

    return {
        "base_dataset": base_dataset,
        "noise_un": noise_un,
        "un_kept": un_datasets["kept"],
        "un_leftover": un_datasets["leftover"],
        "noise_un_synth_targets": noise_un_synth_targets,
        "un_synth_targets_kept": un_synth_datasets["kept"],
        "un_synth_targets_leftover": un_synth_datasets["leftover"],
        "noise_mn": noise_mn,
        "mn_datasets": mn_datasets,  # {"target_cluster": df, "cluster_2": df, ..., "leftover": df}
    }


def main() -> None:
    # OUTPUT_DIRS creation now lives in generate_for_benchmark() itself, so
    # every caller gets it, not just this entry point.
    with open(paths.BENCHMARKS_YAML) as f:
        benchmarks_data = yaml.safe_load(f)
        names = list(benchmarks_data.keys())

    for benchmark_name in names:
        target_column = benchmarks_data[benchmark_name]["target_column"]
        generate_for_benchmark(benchmark_name, target_column)


if __name__ == "__main__":
    main()
