"""
Does the Python LIMA (diada.core.LIMA, the port of the jar; first written here as
explorations/lima_jar_like.py) give the jar's output?

For each benchmark, the CSV DIADA writes for the jar (_label_columns, 10 buckets) is given to both:
  jar       java -Xmx10g -jar DIADA-0.8.jar <csv> 0.000001 <rows>   (as invoke_diada_jar, plus heap)
  python    LIMA(Dataset.from_csv(<csv>)).run()              (same seed 42, same approx)
and the raw outputs (the full column x column matrix) are compared:
  exact        cells with the same value (to 1e-9)
  cells >= 2   cells where both are >= 2 (below 2 the jar's value depends on its hash-map order)
  max diff     largest difference over those cells
  edges        DIADA's edges (soundness > 4) in both / only in one
plus the time of each.

Benchmarks: the old ones (diamonds, bank, house_sales, bank_mn) and, with --noise, the UN and MN
versions of every benchmarks.yaml dataset.

Run with:  python explorations/jar_like_vs_jar.py [--noise] [--only name ...] [--out dir]
Writes results/jar_like_vs_jar/{runs.csv, runs.json}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from diada import paths
from diada.core.diada_tool import _label_columns

from diada.core.LIMA import LIMA, Dataset

OUT = paths.RESULTS_DIR / "jar_like_vs_jar"
# the jar keeps every lattice node it visits; with Java's default heap (1/4 of RAM, ~4 GB here)
# it runs out of memory on mice_protein MN, so it gets more here (DIADA itself calls it without -Xmx)
JAR_HEAP = "10g"
OLD = {"diamonds": "base_datasets/diamonds.csv", "bank": "base_datasets/bank.csv",
       "house_sales": "base_datasets/house_sales.csv", "bank_mn": "noise_mn/bank_mn_noise.csv"}


def jar_matrix(csv: Path, rows: int, cols: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="diada_run_") as tmp:
        cmd = ["java", f"-Xmx{JAR_HEAP}", "-jar", str(paths.JAR_PATH), str(csv), "0.000001", str(rows)]
        subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
        out = pd.read_csv(Path(tmp) / "output.txt", sep=r"\s+", header=None, names=["f", "t", "s"])
    m = np.zeros((cols, cols))
    m[out["f"], out["t"]] = out["s"]
    return m


def edges(m: np.ndarray, threshold: float = 4) -> set:
    s = np.maximum(m, m.T)
    i, j = np.nonzero(np.triu(s > threshold, 1))
    return set(zip(i.tolist(), j.tolist()))


def compare(name: str, rel: str) -> dict:
    df = pd.read_csv(paths.DATA_DIR / rel)
    data = _label_columns(df.copy(), 10)
    with tempfile.TemporaryDirectory(prefix="jarlike_") as tmp:
        csv = Path(tmp) / "temp_data.csv"
        data.to_csv(csv, index=False)
        t0 = time.perf_counter()
        mj = jar_matrix(csv, len(data), data.shape[1])
        t1 = time.perf_counter()
        lima = LIMA(Dataset.from_csv(csv, nrow=len(data)))
        mp, stats = lima.run(), lima.stats
        t2 = time.perf_counter()
    both = (mj >= 2) & (mp >= 2)
    ej, ep = edges(mj), edges(mp)
    return dict(benchmark=name, rows=len(df), cols=df.shape[1], cells=mj.size,
                exact=int(np.isclose(mj, mp, rtol=0, atol=1e-9).sum()),
                cells_ge2_jar=int((mj >= 2).sum()), cells_ge2_py=int((mp >= 2).sum()), cells_ge2_both=int(both.sum()),
                max_diff_ge2=float(np.abs(mj - mp)[both].max()) if both.any() else 0.0,
                edges_jar=len(ej), edges_py=len(ep), edges_both=len(ej & ep), only_jar=len(ej - ep), only_py=len(ep - ej),
                jar_s=round(t1 - t0, 2), py_s=round(t2 - t1, 2), patterns=stats["patterns"],
                pairs_sampled=stats["pairs_sampled"], batches=stats["batches"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    jobs = dict(OLD)
    if args.noise:
        for name in yaml.safe_load(open(paths.REPO_ROOT / "benchmarks.yaml")):
            jobs[f"{name} UN"] = f"noise_un/{name}_un_noise.csv"
            jobs[f"{name} MN"] = f"noise_mn/{name}_mn_noise.csv"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "runs.json"
    runs = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["benchmark"] for r in runs}
    for name, rel in jobs.items():
        if name in done or (args.only and name.split()[0] not in args.only):
            continue
        r = compare(name, rel)
        runs.append(r)
        out_path.write_text(json.dumps(runs, indent=1))
        pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
        print({k: r[k] for k in ("benchmark", "exact", "cells", "max_diff_ge2", "edges_jar", "edges_py", "only_jar",
                                 "only_py", "jar_s", "py_s")}, flush=True)


if __name__ == "__main__":
    main()
