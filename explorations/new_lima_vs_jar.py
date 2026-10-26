"""
Does the corrected Python LIMA (skew correction ported from C:/Projects/LIMA_py, commit e393958) give
graphs closer to the jar's than the old Python LIMA did?

On the old benchmarks (diamonds, bank, house_sales, bank with injected noise):
  jar        DIADA-0.8.jar via invoke_diada(engine="jar"), deterministic, once
  new python invoke_diada(engine="lima") exactly as DIADA calls it, seeds 0, 1, 2
  old python the original Python LIMA before the fix, seeds 0 and 1, from
             results/two_sided/old_runs.json ("V0 base", same labelling and threshold)
Per run: edges (soundness > 4, dataset_builder.build_soundness_graph), edges shared with the jar,
only in the jar, only in Python, overlap (shared / union), DIADA's UN selection and MN target cluster
compared with the jar's, time.

Run with:  python explorations/new_lima_vs_jar.py
Writes results/new_lima_vs_jar/{runs.csv, edges.csv}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from diada import paths
from diada.core.dataset_builder import build_soundness_graph
from diada.core.diada_tool import invoke_diada

sys.path.insert(0, str(Path(__file__).parent))
from two_sided_plots import label  # noqa: E402

OUT = paths.RESULTS_DIR / "new_lima_vs_jar"
OLD = {"diamonds": ("base_datasets/diamonds.csv", "price"), "bank": ("base_datasets/bank.csv", "Class"),
       "house_sales": ("base_datasets/house_sales.csv", "price"), "bank_mn": ("noise_mn/bank_mn_noise.csv", "Class")}
SEEDS = (0, 1, 2)


def edges_of(soundness: pd.DataFrame) -> set:
    return {frozenset(e) for e in build_soundness_graph(soundness, threshold=4).edges}


def summary(name, version, seed, edges, jar, target, seconds) -> dict:
    g, gj = nx.Graph(), nx.Graph()
    g.add_edges_from(tuple(e) for e in edges)
    gj.add_edges_from(tuple(e) for e in jar)
    mn = set(nx.node_connected_component(g, target)) if target in g else {target}
    mn_j = set(nx.node_connected_component(gj, target)) if target in gj else {target}
    return dict(benchmark=name, version=version, seed=seed, edges=len(edges), shared_with_jar=len(edges & jar),
                only_jar=len(jar - edges), only_python=len(edges - jar),
                overlap=round(len(edges & jar) / max(1, len(edges | jar)), 3),
                UN_same_as_jar=set(g.nodes) == set(gj.nodes), MN_size=len(mn), MN_jar=len(mn_j),
                MN_overlap=round(len(mn & mn_j) / len(mn | mn_j), 3), seconds=round(seconds, 2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_runs = {(r["benchmark"], r["seed"]): r for r in json.loads((paths.RESULTS_DIR / "two_sided" / "old_runs.json").read_text())
                if r["version"] == "V0 base"}
    rows, edge_rows = [], []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        start = time.perf_counter()
        jar = edges_of(invoke_diada(df, 10, engine="jar", verbose=False))
        rows.append(summary(name, "jar", 0, jar, jar, target, time.perf_counter() - start))
        found = {}
        for seed in SEEDS:
            np.random.seed(seed)
            start = time.perf_counter()
            e = edges_of(invoke_diada(df, 10, engine="lima", verbose=False))
            rows.append(summary(name, "new python", seed, e, jar, target, time.perf_counter() - start))
            found[seed] = e
            print(name, "new python seed", seed, rows[-1], flush=True)
        for seed in (0, 1):
            r = old_runs[(name, seed)]
            e = {frozenset(k.split("-", 1)) for k in r["edges"]}
            rows.append(summary(name, "old python", seed, e, jar, target, r["seconds"]))
        for e in sorted(jar | set().union(*found.values()), key=lambda e: sorted(e)):
            x, y = sorted(e)
            edge_rows.append(dict(benchmark=name, edge=f"{label(name, x)}-{label(name, y)}", in_jar=e in jar,
                                  new_python_seeds=sum(e in found[s] for s in SEEDS)))
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "edges.csv", index=False)


if __name__ == "__main__":
    main()
