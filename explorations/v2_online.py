"""
V2 inside LIMA ("online V2") vs V2 after LIMA ("post-hoc V2", explorations/v2_mediation_split.py).

Post-hoc: V1 (two-sided LIMA) runs to the end; then every triangle of its graph is tested and the
explained ones are split.

Online: after every LIMA round, every triangle that has just appeared among the column pairs
LIMA has found so far (a pair is found when one of its two-column patterns passes LIMA's test;
bigger patterns never add pairs) is tested with the same test. If B explains A-C, the triangle
is split at once and LIMA stops exploring any pattern that contains both A and C: candidate
edges whose pattern holds both columns are dropped, and new ones are never proposed. The aim is
to stop spending LIMA's samples on patterns built on a link already known to be indirect.

Both use exactly the same test sample (2^20 pairs from their own random generator, so that
drawing it does not shift LIMA's own random draws) and the same splitting rule (best explained
first, only while the triangle still exists). Differences between them come from the pruning and
from the order in which triangles appear.

Cost measured: wall time (LIMA rounds and tests separately), LIMA rounds, pairs sampled, patterns
counted, and LIMA's work = sum over rounds of (patterns scheduled x pairs sampled).

Run with:  python explorations/v2_online.py
Writes results/v2_online/{synthetic.csv, old.csv}
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from diada import paths
from diada.core.dataset_builder import build_soundness_graph
from diada.core.diada_tool import _label_columns, _relatedness_matrix_to_soundness_df

sys.path.insert(0, str(Path(__file__).parent))
from lima_two_sided import TwoSidedLatticeState, TwoSidedLIMA  # noqa: E402
from synthetic_triads import EXPECTED  # noqa: E402
from v2_mediation_split import N_PAIRS, PairSample, find_splits, mediation_test, split_groups  # noqa: E402

OUT = paths.RESULTS_DIR / "v2_online"
OLD = {"diamonds": "base_datasets/diamonds.csv", "bank": "base_datasets/bank.csv",
       "house_sales": "base_datasets/house_sales.csv", "bank_mn": "noise_mn/bank_mn_noise.csv"}


class PruningState(TwoSidedLatticeState):
    """Two-sided LIMA state that never explores a pattern holding both columns of a split pair."""

    def __init__(self, lattice):
        super().__init__(lattice, two_sided=True)
        self.split_pgs: list = []

    def blocked(self, node) -> bool:
        return any(a in node.preds and c in node.preds for a, c in self.split_pgs)

    def _isCandidate(self, edge):
        return not self.blocked(edge.to()) and super()._isCandidate(edge)

    def prune(self, pg_a: int, pg_c: int) -> int:
        self.split_pgs.append((pg_a, pg_c))
        before = len(self.candidateEdges)
        self.candidateEdges = {e for e in self.candidateEdges if not self.blocked(e.to())}
        return before - len(self.candidateEdges)


class MeasuredLIMA(TwoSidedLIMA):
    """Two-sided LIMA (V1) that records its work and, if online, runs V2's test after every round."""

    def __init__(self, dataset, online: bool, test_pairs: np.ndarray):
        super().__init__(dataset)
        self.state = PruningState(self.lattice)
        self.online = online
        self.names = [c.split("(")[0] for c in self.dataset.columns]
        self.pg = {n: i for i, n in enumerate(self.names)}
        self.test_pairs = test_pairs
        self.sample = None
        self.work = 0
        self.lima_seconds = 0.0
        self.test_seconds = 0.0
        self.tested: set = set()
        self.tests: list = []
        self.splits: list = []
        self.dropped_candidates = 0

    def step(self, iteration: int) -> bool:
        start = time.perf_counter()
        sched = self.scheduler.schedule(self.state, self.approx)
        if not sched.parents:
            self.lima_seconds += time.perf_counter() - start
            return True
        n = self.getStepSize(iteration)
        result = self.sampler.sample(sched, self.domain, self.predicate_groups, self.state, n)
        self.state.update(result.counts, self.devs)
        self.work += len(sched.parents) * n
        self.lima_seconds += time.perf_counter() - start
        if self.online:
            start = time.perf_counter()
            self._split_new_triangles()
            self.test_seconds += time.perf_counter() - start
        return False

    def pair_graph(self) -> nx.Graph:
        """Column pairs found so far: pairs with a sound two-column pattern."""
        g = nx.Graph()
        for e in self.state.soundEdges:
            if len(e.fr().preds) == 1:
                (x,) = e.fr().preds
                g.add_edge(self.names[x], self.names[e.pg])
        return g

    def get_sample(self) -> PairSample:
        if self.sample is None:
            self.sample = PairSample(self, pairs=self.test_pairs)
        return self.sample

    def _split_new_triangles(self) -> None:
        g = self.pair_graph()
        g.remove_edges_from((t["a"], t["c"]) for t in self.splits)
        new = []
        for a, c in g.edges:
            for b in nx.common_neighbors(g, a, c):
                key = (frozenset((a, c)), b)
                if key not in self.tested:
                    self.tested.add(key)
                    new.append(mediation_test(self.get_sample(), a, b, c))
        self.tests += new
        for t in sorted((t for t in new if t["explained"]), key=lambda t: t["share_upper"]):
            a, b, c = t["a"], t["b"], t["c"]
            if g.has_edge(a, c) and g.has_edge(a, b) and g.has_edge(b, c):
                g.remove_edge(a, c)
                self.splits.append(t)
                self.dropped_candidates += self.state.prune(self.pg[a], self.pg[c])


def test_pairs(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    p = rng.integers(0, n_rows, size=(N_PAIRS, 2)).astype(np.int32)
    same = p[:, 0] == p[:, 1]
    p[same, 1] = (p[same, 1] + 1) % n_rows
    return p


def run_mode(df: pd.DataFrame, mode: str, seed: int) -> dict:
    """mode: "V1" (LIMA only), "post-hoc" (V1, then V2 on its final graph), "online"."""
    labelled = _label_columns(df.copy(), 10)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "input.csv"
        labelled.to_csv(path, index=False)
        np.random.seed(seed)
        lima = MeasuredLIMA(str(path), online=mode == "online", test_pairs=test_pairs(len(df), seed))
        rounds = 0
        for i in range(100):
            if lima.step(i):
                break
            rounds = i + 1
    graph = build_soundness_graph(_relatedness_matrix_to_soundness_df(lima.relatedness(), labelled.columns.tolist()), threshold=4)
    splits, tests = lima.splits, lima.tests
    if mode == "post-hoc":
        start = time.perf_counter()
        _, splits, tests = find_splits(graph, lima.get_sample())
        lima.test_seconds = time.perf_counter() - start
    final = graph.copy()
    final.remove_edges_from((t["a"], t["c"]) for t in splits)
    groups = split_groups(final, {frozenset((t["a"], t["c"])) for t in splits}) if mode != "V1" else None
    return dict(mode=mode, seed=seed, rounds=rounds, pairs_sampled=sum(lima.getStepSize(i) for i in range(rounds)),
                patterns_counted=len(lima.state._dists), sound_edges=len(lima.state.soundEdges), work=lima.work,
                lima_seconds=round(lima.lima_seconds, 3), test_seconds=round(lima.test_seconds, 3),
                total_seconds=round(lima.lima_seconds + lima.test_seconds, 3), edges=graph.number_of_edges(),
                triangles_tested=len(tests), splits=len(splits), dropped_candidates=lima.dropped_candidates,
                split_pairs=" ".join(sorted("-".join(sorted((t["a"], t["c"]))) for t in splits)),
                edge_set=" ".join(sorted("-".join(sorted(e)) for e in graph.edges)),
                groups=" | ".join("-".join(g) for g in groups) if groups is not None else "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in EXPECTED:
        df = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{name}.csv")
        for seed in range(5):
            for mode in ("V1", "post-hoc", "online"):
                rows.append(dict(benchmark=name, **run_mode(df, mode, seed)))
        print(f"synthetic {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "synthetic.csv", index=False)
    rows = []
    for name, rel in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        for seed in (0, 1):
            for mode in ("V1", "post-hoc", "online"):
                rows.append(dict(benchmark=name, **run_mode(df, mode, seed)))
                print(name, seed, mode, {k: rows[-1][k] for k in ("total_seconds", "work", "splits", "edges")}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "old.csv", index=False)


if __name__ == "__main__":
    main()
