"""
Jar vs. the corrected Python LIMA on DIADA's original task (experiment 3): cleaning benchmarks with
injected noise.

  UN version: the base table + pure-noise columns (noise_*) + fully reshuffled copies of every
              feature (*_shuffled_*). All injected columns are unrelated to everything, so DIADA's
              UN step (keep the columns with at least one edge) should drop all of them.
  MN version: the UN table + a cluster of mutually correlated spurious columns (spurious_*),
              unrelated to the original data. DIADA's MN step keeps the target's connected group,
              which should contain no injected column.

For every benchmark of benchmarks.yaml, both versions: the jar once (deterministic) and the Python
LIMA as DIADA calls it (engine="lima") with seeds 0 and 1. Per run: the edges (soundness > 4), the
UN selection, the MN target group, time. Saved after every run, smallest benchmarks first.

Run with:  python explorations/jar_vs_fixed_lima_diada.py [--only name ...]
Writes results/jar_vs_fixed_lima/runs.json
"""

from __future__ import annotations

import argparse
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from diada import paths
from diada.core.dataset_builder import build_soundness_graph
from diada.core.diada_tool import invoke_diada

OUT = paths.RESULTS_DIR / "jar_vs_fixed_lima"
SEEDS = (0, 1)


def kind(col: str) -> str:
    if col.startswith("noise_"):
        return "noise"
    if "_shuffled_" in col:
        return "shuffled"
    if col.startswith("spurious_"):
        return "spurious"
    return "original"


def describe(df: pd.DataFrame, soundness: pd.DataFrame, target: str) -> dict:
    g = build_soundness_graph(soundness, threshold=4)
    mn = sorted(nx.node_connected_component(g, target)) if target in g else [target]
    return dict(edges=sorted("|".join(sorted(e)) for e in g.edges), un_kept=sorted(g.nodes), mn_target=mn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "runs.json"
    runs = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["benchmark"], r["variant"], r["engine"], r["seed"]) for r in runs}
    config = yaml.safe_load(open(paths.REPO_ROOT / "benchmarks.yaml"))
    jobs = []
    for name, cfg in config.items():
        if args.only and name not in args.only:
            continue
        for variant, folder, suffix in (("UN", "noise_un", "un_noise"), ("MN", "noise_mn", "mn_noise")):
            path = paths.DATA_DIR / folder / f"{name}_{suffix}.csv"
            with open(path) as fh:
                n_cols = len(fh.readline().split(","))
            jobs.append((n_cols, name, variant, path, cfg["target_column"]))
    for n_cols, name, variant, path, target in sorted(jobs):
        df = pd.read_csv(path)
        for engine, seeds in (("jar", (0,)), ("lima", SEEDS)):
            for seed in seeds:
                if (name, variant, engine, seed) in done:
                    continue
                np.random.seed(seed)
                start = time.perf_counter()
                soundness = invoke_diada(df, 10, engine=engine, verbose=False)
                seconds = time.perf_counter() - start
                runs.append(dict(benchmark=name, variant=variant, engine=engine, seed=seed, rows=len(df), cols=df.shape[1],
                                 target=target, seconds=round(seconds, 2), **describe(df, soundness, target)))
                out_path.write_text(json.dumps(runs))
                print(f"{name} {variant} {engine} seed {seed}: {len(runs[-1]['edges'])} edges, {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
