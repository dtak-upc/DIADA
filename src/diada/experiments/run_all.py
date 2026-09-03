"""
Run any/all of the numbered reproducibility experiments in one invocation.

Each experiment lives in its own subpackage and is independently runnable
(see EXPERIMENTS.md for what each one measures and its own module for full
details):

    1  experiment_1_scalability.scalability_benchmark  -- LIMA vs. Apriori/FP-Growth
    2  experiment_2_ufs.ufs_benchmark                   -- DIADA vs. UFS baselines
    3  experiment_3_diada_arda.run_paper_experiments    -- DIADA+ARDA on every
                                                             benchmark x noise variant

This script just sequences their existing `main()` entry points -- no
experiment logic lives here. Every experiment is independently resumable/
checkpointed (see each module's own docstring), so interrupting a run
started here and re-running is safe.

All three are expensive (each can run for hours on the full benchmark
suite); nothing here shortcuts that. Select a subset with --experiments if
you don't want to run everything.

Run all three:        python -m diada.experiments.run_all
Run a subset:          python -m diada.experiments.run_all --experiments 1 2
Pass Experiment 3 args (the only one with its own CLI options) after --:
                       python -m diada.experiments.run_all --experiments 3 -- --datasets diamonds --max-features 5
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

EXPERIMENTS = {
    1: ("Experiment 1: LIMA scalability vs. Apriori/FP-Growth", "experiment_1_scalability"),
    2: ("Experiment 2: DIADA vs. UFS baselines", "experiment_2_ufs"),
    3: ("Experiment 3: DIADA+ARDA on every benchmark x noise variant", "experiment_3_diada_arda"),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_experiment_1() -> None:
    from .experiment_1_scalability import scalability_benchmark
    scalability_benchmark.main()


def run_experiment_2() -> None:
    from .experiment_2_ufs import ufs_benchmark
    ufs_benchmark.main()


def run_experiment_3(argv: Optional[List[str]] = None) -> None:
    from .experiment_3_diada_arda import run_paper_experiments
    run_paper_experiments.main(argv)


_RUNNERS = {1: run_experiment_1, 2: run_experiment_2, 3: run_experiment_3}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m diada.experiments.run_all",
        description="Run any/all of the numbered reproducibility experiments in sequence.",
    )
    parser.add_argument("--experiments", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
                         help="Which experiment(s) to run, by number (default: all three, in order).")
    parser.add_argument("experiment_3_args", nargs=argparse.REMAINDER,
                         help="Everything after '--' is forwarded to Experiment 3's own argument "
                              "parser (--datasets, --max-features, --output-dir, --quiet). Ignored "
                              "if Experiment 3 isn't selected.")
    args = parser.parse_args(argv)

    exp3_args = args.experiment_3_args
    if exp3_args and exp3_args[0] == "--":
        exp3_args = exp3_args[1:]

    _log(f"Running experiment(s) {args.experiments}")
    for n in args.experiments:
        title, _ = EXPERIMENTS[n]
        _log(f"=== {title} ===")
        if n == 3:
            _RUNNERS[n](exp3_args or None)
        else:
            _RUNNERS[n]()
        _log(f"=== {title} -- done ===")

    _log("ALL SELECTED EXPERIMENTS DONE")


if __name__ == "__main__":
    main()
