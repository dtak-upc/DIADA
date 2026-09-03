import math
import pickle
import time
from typing import Dict, List, Tuple

from data.dataset.Dataset import Dataset
from LIMA.LIMA import LIMA
from LIMA.sampler.APRIORISampler import APRIORISampler
from LIMA.sampler.FPSampler import FPSampler
from LIMA.sampler.LIMASampler import LIMASampler
from LIMA.scheduler.LIMAScheduler import LIMAScheduler
from LIMA.scheduler.MonoScheduler import MonoScheduler

DATASET_PATHS = ["miniboone.csv","jannis.csv","drive_diagnosis.csv"]
APPROX_FACTORS = [1e-5, 1e-7, 1e-9]
ALGORITHMS = ( "apriori","FP-GROWTH","LIMA")

APRIORI_FP_DEPTH = 3  
REPEATS = 5  

RESULTS_PATH = "benchmark_results.pkl"
SCALABILITY_RESULTS_PATH = "scalability_results.pkl"
SAMPLE_SIZE_SCALABILITY_RESULTS_PATH = "sample_size_scalability_results.pkl"
COMBINED_RESULTS_PATH = "results2.pickle"
COLUMN_SCALABILITY_APPROX = 1e-5  
TIME_LIMIT_SECONDS = 1800.0

APROX_INV_MIN = 1e3
APROX_INV_MAX = 1e8
APROX_INV_POINTS = 40  


def _deriveParams(approx: float, dataset_size: int) -> Tuple[int, int]:
    """estN = 1/approx, capped at 0.0125 * dataset_size**2 (the number of
    distinct ordered pairs is on the order of dataset_size**2, so this keeps
    estN from growing unboundedly past what the dataset can actually support);
    max_iter = 100 safety, never reached
    """
    estN = 10.0 / approx
    cap = 1 * dataset_size**2
    estN = min(estN, cap)
    max_iter = 100
    return int(estN), max_iter


def _timeRun(lima: LIMA, step_size: int, max_iter: int) -> float:
    start = time.perf_counter()
    for i in range(max_iter):
        if lima.step(i):
            break
    return time.perf_counter() - start


def _buildLima(dataset, approx: float, algo: str, dataset_size: int) -> Tuple[LIMA, int]:
    """Instantiate LIMA (scheduler/sampler per `algo`) and the step size it
    should be run with."""
    if algo == "LIMA":
        lima = LIMA(dataset, scheduler=LIMAScheduler(), sampler=LIMASampler(), approx=approx)
        return lima, 100

    estN, _ = _deriveParams(approx, dataset_size)
    mono_n = estN * 2
    if algo == "apriori":
        sampler = APRIORISampler(APRIORI_FP_DEPTH)
    elif algo == "FP-GROWTH":
        sampler = FPSampler(APRIORI_FP_DEPTH)
    else:
        raise ValueError(f"unknown algorithm {algo!r}")

    lima = LIMA(dataset, scheduler=MonoScheduler(), sampler=sampler, approx=approx)
    return lima, mono_n


def _runAlgo(dataset, approx: float, algo: str) -> float:
    """Build a fresh LIMA for `algo` over `dataset` and time it to completion,
    repeating REPEATS times and returning the average elapsed seconds."""
    dataset_obj = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
    dataset_size = len(dataset_obj.df)

    _, max_iter = _deriveParams(approx, dataset_size)

    elapsed_times = []
    for _ in range(REPEATS):
        lima, step_size = _buildLima(dataset_obj, approx, algo, dataset_size)
        elapsed_times.append(_timeRun(lima, step_size, max_iter))
    return sum(elapsed_times) / len(elapsed_times)


def runBenchmark(
    dataset_paths: List[str], approxs: List[float]
) -> Dict[Tuple[str, float, str], float]:
    """Time LIMA/apriori/FP-GROWTH on every (dataset, approx) pair.

    Returns a dict keyed by (dataset_path, approx, algorithm) -> elapsed
    seconds, algorithm in {"LIMA", "apriori", "FP-GROWTH"}.
    """
    results: Dict[Tuple[str, float, str], float] = {}

    for dataset_path in dataset_paths:
        for approx in approxs:
            for algo in ALGORITHMS:
                print(f"[{dataset_path} | approx={approx} | {algo}] running fixed-size benchmark...")
                elapsed = _runAlgo(dataset_path, approx, algo)
                results[(dataset_path, approx, algo)] = elapsed
                print(
                    f"[{dataset_path} | approx={approx} | {algo}] done in {elapsed:.3f}s"
                )

    return results


# ---------------------------------------------------------------------------
# Scalability benchmark: time each configuration as rows/columns grow.
# ---------------------------------------------------------------------------

def _rowsSeries() -> List[int]:
    """10k..100k in steps of 10k, then 200k..1M in steps of 100k."""
    return list(range(10_000, 100_000 + 1, 10_000)) + list(range(200_000, 1_000_000 + 1, 100_000))


def _colsSeries() -> List[int]:
    """10..60 in steps of 2."""
    return list(range(10, 60 + 1, 2))


_BASE_COLS = 30  # columns held fixed while the rows series grows - set manually
_BASE_ROWS = 20_000  # rows held fixed while the columns series grows - set manually


def _runSeries(
    base_dataset: Dataset,
    approx: float,
    algo: str,
    sizes: List[int],
    fixed_other: int,
    vary_rows: bool,
    dataset_path: str,
    series_name: str,
) -> List[Tuple[int, float]]:
    """Time `algo` over `base_dataset.expand(...)` at each size in `sizes`
    (rows if `vary_rows` else columns, with the other dimension held at
    `fixed_other`), stopping as soon as a run exceeds TIME_LIMIT_SECONDS.
    """
    series: List[Tuple[int, float]] = []
    for size in sizes:
        rows, cols = (size, fixed_other) if vary_rows else (fixed_other, size)
        label = f"[{dataset_path} | approx={approx} | {algo} | {series_name}] rows={rows} cols={cols}"
        print(f"{label} running...")

        expanded = base_dataset.expand(rows, cols)
        elapsed = _runAlgo(expanded, approx, algo)
        series.append((size, elapsed))

        print(f"{label} done in {elapsed:.3f}s")

        if elapsed > TIME_LIMIT_SECONDS:
            print(
                f"[{dataset_path} | approx={approx} | {algo} | {series_name}] "
                f"{elapsed:.3f}s exceeded the {TIME_LIMIT_SECONDS:.0f}s limit, stopping series here"
            )
            break
    return series


def runScalabilityBenchmark(
    dataset_paths: List[str], approxs: List[float]
) -> Dict[Tuple[str, float, str], Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]:
    """For every (dataset, approx, algorithm), times two growth series:
    increasing row count (columns fixed at `_BASE_COLS`) and increasing
    column count (rows fixed at `_BASE_ROWS`). A series stops growing as
    soon as one of its runs exceeds TIME_LIMIT_SECONDS.

    Returns {(dataset_path, approx, algorithm): (rows_series, cols_series)},
    each series a list of (size, elapsed_seconds) pairs.
    """
    results: Dict[
        Tuple[str, float, str], Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]
    ] = {}

    for dataset_path in dataset_paths:
        print(f"[{dataset_path}] loading base dataset...")
        base_dataset = Dataset(dataset_path)

        for approx in approxs:
            for algo in ALGORITHMS:
                print(
                    f"[{dataset_path} | approx={approx} | {algo}] "
                    f"starting rows series ({_rowsSeries()[0]}..{_rowsSeries()[-1]}, "
                    f"cols fixed at {_BASE_COLS})"
                )
                rows_series = _runSeries(
                    base_dataset,
                    approx,
                    algo,
                    _rowsSeries(),
                    _BASE_COLS,
                    vary_rows=True,
                    dataset_path=dataset_path,
                    series_name="rows",
                )
                print(
                    f"[{dataset_path} | approx={approx} | {algo}] "
                    f"rows series finished, {len(rows_series)} point(s): {rows_series}"
                )

                print(
                    f"[{dataset_path} | approx={approx} | {algo}] "
                    f"starting columns series ({_colsSeries()[0]}..{_colsSeries()[-1]}, "
                    f"rows fixed at {_BASE_ROWS})"
                )
                cols_series = _runSeries(
                    base_dataset,
                    approx,
                    algo,
                    _colsSeries(),
                    _BASE_ROWS,
                    vary_rows=False,
                    dataset_path=dataset_path,
                    series_name="cols",
                )
                print(
                    f"[{dataset_path} | approx={approx} | {algo}] "
                    f"columns series finished, {len(cols_series)} point(s): {cols_series}"
                )

                results[(dataset_path, approx, algo)] = (rows_series, cols_series)

    return results


# ---------------------------------------------------------------------------
# Sample-size scalability benchmark: like the rows series, but instead of
# growing the dataset itself, grows the sample size by sweeping aproxInv
# (approx = 1/aproxInv) over the *original*, unmodified dataset.
# ---------------------------------------------------------------------------

def _aproxInvSeries() -> List[float]:
    """Logarithmically spaced aproxInv values from APROX_INV_MIN to APROX_INV_MAX."""
    if APROX_INV_POINTS == 1:
        return [APROX_INV_MIN]
    log_min = math.log10(APROX_INV_MIN)
    log_max = math.log10(APROX_INV_MAX)
    step = (log_max - log_min) / (APROX_INV_POINTS - 1)
    return [10 ** (log_min + i * step) for i in range(APROX_INV_POINTS)]


def _runSampleSizeSeries(
    base_dataset: Dataset,
    algo: str,
    aprox_invs: List[float],
    dataset_path: str,
) -> List[Tuple[float, float]]:
    """Time `algo` on the original `base_dataset` (never expanded) for each
    aproxInv in `aprox_invs`, approx = 1/aproxInv; the params for the other
    algorithms are still estimated via `_deriveParams`/`_buildLima` as
    usual. Stops as soon as a run exceeds TIME_LIMIT_SECONDS.
    """
    series: List[Tuple[float, float]] = []
    for aprox_inv in aprox_invs:
        approx = 1.0 / aprox_inv
        label = f"[{dataset_path} | {algo} | sampleSize] aproxInv={aprox_inv:.3g} (approx={approx:.3g})"
        print(f"{label} running...")

        elapsed = _runAlgo(base_dataset, approx, algo)
        series.append((aprox_inv, elapsed))

        print(f"{label} done in {elapsed:.3f}s")

        if elapsed > TIME_LIMIT_SECONDS:
            print(
                f"[{dataset_path} | {algo} | sampleSize] "
                f"{elapsed:.3f}s exceeded the {TIME_LIMIT_SECONDS:.0f}s limit, stopping series here"
            )
            break
    return series


def runSampleSizeScalabilityBenchmark(
    dataset_paths: List[str],
) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """For every (dataset, algorithm), times a run on the original dataset
    for each aproxInv in a linear APROX_INV_MIN..APROX_INV_MAX sweep - a
    row-scalability test that grows the effective sample size instead of
    the dataset itself. Stops a series early once a run exceeds
    TIME_LIMIT_SECONDS.

    Returns {(dataset_path, algorithm): [(aproxInv, elapsed_seconds), ...]}.
    """
    results: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    aprox_invs = _aproxInvSeries()

    for dataset_path in dataset_paths:
        print(f"[{dataset_path}] loading base dataset...")
        base_dataset = Dataset(dataset_path)

        for algo in ALGORITHMS:
            print(
                f"[{dataset_path} | {algo}] starting sample-size series "
                f"(aproxInv {aprox_invs[0]:.3g}..{aprox_invs[-1]:.3g})"
            )
            series = _runSampleSizeSeries(base_dataset, algo, aprox_invs, dataset_path)
            print(
                f"[{dataset_path} | {algo}] sample-size series finished, "
                f"{len(series)} point(s): {series}"
            )
            results[(dataset_path, algo)] = series

    return results


def main() -> None:
    row_scalability = runSampleSizeScalabilityBenchmark(DATASET_PATHS)
    scalability = runScalabilityBenchmark(DATASET_PATHS, [COLUMN_SCALABILITY_APPROX])

    combined: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for dataset_path in DATASET_PATHS:
        combined[dataset_path] = {}
        for algo in ALGORITHMS:
            row_series = row_scalability[(dataset_path, algo)]
            _, col_series = scalability[(dataset_path, COLUMN_SCALABILITY_APPROX, algo)]
            combined[dataset_path][algo] = {
                "row": ([x for x, _ in row_series], [y for _, y in row_series]),
                "col": ([x for x, _ in col_series], [y for _, y in col_series]),
            }

    with open(COMBINED_RESULTS_PATH, "wb") as f:
        pickle.dump(combined, f)
    for dataset_path, algos in combined.items():
        for algo, series in algos.items():
            print(f"{(dataset_path, algo)}: {series}")


if __name__ == "__main__":
    main()
