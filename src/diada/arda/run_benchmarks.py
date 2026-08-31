"""
Sweep every benchmark in benchmarks.yaml through ARDA's primary,
feature-selection-only pipeline (ARDA.fit_transform_full_table) -- kept for
reproducibility. Was arda_improved/main.py.

Run with:  python -m diada.arda.run_benchmarks
"""

import pandas as pd
import yaml

from .. import paths
from .arda import ARDA


def main() -> None:
    with open(paths.BENCHMARKS_YAML) as f:
        benchmarks_data = yaml.safe_load(f)
        names = list(benchmarks_data.keys())

    for benchmark_name in names:
        benchmark_configuration = benchmarks_data[benchmark_name]

        target_column = benchmark_configuration["target_column"]
        task = benchmark_configuration["task"]
        metric = benchmark_configuration["metric"]

        benchmark_path = paths.BASE_DATASETS_DIR / f"{benchmark_name}.csv"
        base_dataset = pd.read_csv(benchmark_path)

        arda = ARDA(target_col=target_column, coreset_size=None, coreset_method="stratified" if task == "classification" else "uniform",
                    task=task, metric=metric, rifs_k=10, rifs_eta=0.2, cv=3, verbose=True, benchmark=benchmark_name)

        arda.fit_transform_full_table(base_dataset, target_column, task, metric, max_features=5)


if __name__ == "__main__":
    main()
