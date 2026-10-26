"""
Baseline: the PC algorithm (causal-learn, PC-stable) limited to level 1 (conditioning sets of at
most one column), on the same 10-bucket codes DIADA gives LIMA, with the G-test for categorical
data. Compared with V0 (jar), V1 (two-sided LIMA) and V2 (V1 + triangle splits) from
results/v2/{synthetic.csv, old.json}.

Per benchmark and significance level alpha: total time, time of level 0 alone (max_k=0),
number of independence tests, edges after level 0 and after level 1, and colliders found.

causal-learn is not installed in this environment: run with its folder on PYTHONPATH, e.g.
    PYTHONPATH=<scratchpad>/pylib python explorations/pc_baseline.py
Writes results/pc_baseline/runs.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402
from synthetic_triads import EXPECTED  # noqa: E402

OUT = paths.RESULTS_DIR / "pc_baseline"
OLD = {"diamonds": "base_datasets/diamonds.csv", "bank": "base_datasets/bank.csv",
       "house_sales": "base_datasets/house_sales.csv", "bank_mn": "noise_mn/bank_mn_noise.csv"}
ALPHAS = [0.05, 1e-4]


def skeleton(cg) -> set:
    g = cg.G.graph
    n = g.shape[0]
    return {(i, j) for i in range(n) for j in range(i + 1, n) if g[i, j] != 0 or g[j, i] != 0}


def colliders(cg) -> int:
    """Edges oriented into a node from both sides (i -> k <- j) in PC's output."""
    g = cg.G.graph  # causal-learn: g[j, i] == 1 and g[i, j] == -1 means i -> j
    n = g.shape[0]
    count = 0
    for k in range(n):
        parents = [i for i in range(n) if g[k, i] == 1 and g[i, k] == -1]
        count += sum(1 for a in range(len(parents)) for b in range(a + 1, len(parents))
                     if g[parents[a], parents[b]] == 0)
    return count


def run_pc(data: np.ndarray, alpha: float) -> dict:
    t0 = time.perf_counter()
    cg0 = pc(data, alpha, "gsq", stable=True, max_k=0, show_progress=False)
    t1 = time.perf_counter()
    cg1 = pc(data, alpha, "gsq", stable=True, max_k=1, show_progress=False)
    t2 = time.perf_counter()
    return dict(seconds_level0=round(t1 - t0, 3), seconds_level1_total=round(t2 - t1, 3),
                tests=len(cg1.test.pvalue_cache), edges_level0=len(skeleton(cg0)), edges_level1=len(skeleton(cg1)),
                colliders=colliders(cg1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sets = [(n, paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{n}.csv") for n in EXPECTED] + \
           [(n, paths.DATA_DIR / rel) for n, rel in OLD.items()]
    for name, path in sets:
        df = pd.read_csv(path)
        codes = bucket_codes(df)
        data = np.column_stack([codes[c] for c in df.columns]).astype(float)
        for alpha in ALPHAS:
            r = run_pc(data, alpha)
            rows.append(dict(benchmark=name, rows=len(df), cols=df.shape[1], alpha=alpha, **r))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)


if __name__ == "__main__":
    main()
