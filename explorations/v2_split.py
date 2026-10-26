"""
V2 = V1 (explorations/v1_two_sided.py: the port of the jar with the two-sided test) + splitting
the triangles whose A-C link is mediated by B.

The same V2 as explorations/v2_mediation_split.py (which ran on the old Python LIMA, now replaced),
rebuilt on the new engine. The method is unchanged; only what it draws on changed:
    pairs of rows    a fresh sample of N_PAIRS pairs from the engine's sampler (Java's
                     SplittableRandom, seeded with the run's seed xor TEST_STREAM, so it is not
                     the sample LIMA's graph was built on)
    predicates       the two rows agree / differ on a column, on the engine's column codes
    statistic        m and its variance as the old LIMA's BetaDist computed them: for a pattern
                     seen in k of n pairs, Beta(1 + k, 1 + n - k) and the log of its share,
                     E[log p] = digamma(a) - digamma(a + b), Var[log p] = trigamma(a) -
                     trigamma(a + b) (not log-odds, whatever the old field names said), now with
                     the engine's digamma/trigamma (the jar's approximations)

After V1 has built DIADA's graph (an edge where the soundness is above 4):

1. Triangles. Every edge A-C with every column B linked to both A and C.

2. The test "does B explain A-C?". For each pattern (A agree/differ, C agree/differ):
       m = [log P(A, B, C) - log P(A, B)] - [log P(B, C) - log P(B)]
   (logs of the shares of sampled pairs, as the old nodeDist4 did), i.e. how much the A pattern
   changes the share of pairs with the C pattern, among the pairs whose two rows have the same
   value v of B, for every value v of B separately (pairs with different values of B are not
   used: there B's own effect shows up as an A-C link, and pooling the values of B can hide one).

   The reference is the A-C link when B is ignored: the same m without B, over all pairs:
       m_AC = log P(C | A) - log P(C),   size of the link = largest |m_AC| over the patterns.

   The values of B are combined per pattern: the average of m weighted by precision (1 / variance),
   with variance 1 / (sum of the weights).

   Equivalence test ("is the effect given B surely small?"): B explains A-C if, for every
   pattern,
       |m| + 4 * sd(m)  <  0.2 * size of the A-C link
   Too few pairs give a wide range and no split, so "unsure" means keep.

3. Applying the splits. Triangles are taken from the best explained (smallest upper bound
   relative to the link) up; a split removes A-C and is only applied while the triangle still
   exists, so near-duplicate columns (each edge explained by the third) end up as a chain instead
   of falling apart.

4. Grouping. From the graph without the split edges, with the split pairs (A, C) as pairs that may
   not share a group: a group is a largest set of columns connected through the remaining edges
   with no split pair in it. A split triangle A-B-C gives {A, B} and {B, C}; a triangle that is not
   split and an open path A-B-C stay one group; with no split the groups are DIADA's connected
   components. For MN, the target's dataset is the union of the groups that contain the target.

Nothing under src/diada/ is modified.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Set

import networkx as nx
import numpy as np

from diada.core.LIMA.java_random import JavaSplittableRandom
from diada.core.LIMA.lima import y1, y2

N_PAIRS = 1 << 20  # pairs sampled for the tests (LIMA draws ~1-3 million pairs in a whole run)
DEVS = 4.0  # LIMA's confidence level
EXPLAINED = 0.8  # B must explain at least 80% of the A-C link
PATTERNS = [(True, True), (True, False), (False, True), (False, False)]  # (A agrees, C agrees)
TEST_STREAM = 0x5DEECE66D  # the test sample's seed is the run's seed xor this


def _log_share(k: float, n: float) -> tuple:
    """E[log p] and Var[log p] for p ~ Beta(1 + k, 1 + n - k): a pattern seen in k of n pairs"""
    a, ab = 1 + int(k), 2 + int(n)
    return y1(a) - y1(ab), y2(a) - y2(ab)


def lima_m(from1: float, to1: float, from2: float, to2: float, n: float):
    """The old LIMA's comparison (betaDist.nodeDist4): m and its variance, from pair counts out of
    n sampled pairs."""
    f1, t1, f2, t2 = _log_share(from1, n), _log_share(to1, n), _log_share(from2, n), _log_share(to2, n)
    return (t1[0] - f1[0]) - (t2[0] - f2[0]), t1[1] + f1[1] + t2[1] + f2[1]


class PairSample:
    """One sample of row pairs drawn with the engine's sampler; per column, whether each pair agrees
    and the value of the pair's first row. Counts are cached: per column pair, the four
    agree/differ patterns of every sampled pair; per column, the pairs that agree on it and their
    value."""

    def __init__(self, lima, n_pairs: int = N_PAIRS):
        ds = lima.dataset
        self.n = n_pairs
        self._x, self._y = JavaSplittableRandom(lima.seed ^ TEST_STREAM).pairs(n_pairs, ds.size)
        self._values = {c.split("(")[0]: v for c, v in zip(ds.columns, ds.values)}
        self._agree: Dict[str, np.ndarray] = {}
        self._on: Dict[str, tuple] = {}
        self._pattern: Dict[tuple, np.ndarray] = {}
        self._link: Dict[frozenset, float] = {}

    def agree(self, col: str) -> np.ndarray:
        if col not in self._agree:
            v = self._values[col]
            self._agree[col] = v[self._x] == v[self._y]
        return self._agree[col]

    def on(self, col: str):
        """(indices of the pairs whose two rows agree on col, their value of col)."""
        if col not in self._on:
            idx = np.flatnonzero(self.agree(col))
            self._on[col] = idx, self._values[col][self._x[idx]].astype(np.int64)
        return self._on[col]

    def pattern(self, a: str, c: str) -> np.ndarray:
        """Per sampled pair, the index in PATTERNS of its (A agrees, C agrees) pattern."""
        if (a, c) not in self._pattern:
            self._pattern[(a, c)] = 2 * (~self.agree(a)).astype(np.int8) + (~self.agree(c)).astype(np.int8)
        return self._pattern[(a, c)]

    def link(self, a: str, c: str) -> float:
        """Size of the A-C link when everything else is ignored: the largest |m| over the four
        patterns, m = log P(C pattern | A pattern) - log P(C pattern)."""
        key = frozenset((a, c))
        if key not in self._link:
            k = np.bincount(self.pattern(a, c), minlength=4).astype(float)
            sizes = []
            for i, (pa, pc) in enumerate(PATTERNS):
                a_side = sum(k[j] for j, p in enumerate(PATTERNS) if p[0] == pa)
                c_side = sum(k[j] for j, p in enumerate(PATTERNS) if p[1] == pc)
                sizes.append(abs(lima_m(a_side, k[i], self.n, c_side, self.n)[0]))
            self._link[key] = max(sizes)
        return self._link[key]


def effect_given(s: PairSample, a: str, b: str, c: str) -> dict:
    """How A's pattern changes C's pattern among pairs with the same value of B: per pattern, the
    m of every value of B combined with precision weights (1 / variance); for the pattern with the
    largest upper bound |m| + DEVS * sd: that bound, m, and its z = m / sd."""
    idx, val = s.on(b)
    nb = int(val.max()) + 1 if len(val) else 1
    counts = np.bincount(val * 4 + s.pattern(a, c)[idx], minlength=nb * 4).reshape(nb, 4).astype(float)
    present = [v for v in range(nb) if counts[v].sum() > 0]
    worst, worst_m, worst_z, worst_at = 0.0, 0.0, 0.0, None
    for i, (pa, pc) in enumerate(PATTERNS):
        ms, ws = [], []
        for v in present:
            from1 = counts[v, [j for j, p in enumerate(PATTERNS) if p[0] == pa]].sum()
            to2 = counts[v, [j for j, p in enumerate(PATTERNS) if p[1] == pc]].sum()
            m, var = lima_m(from1, counts[v, i], counts[v].sum(), to2, s.n)
            ms.append(m)
            ws.append(1.0 / var)
        ms, ws = np.array(ms), np.array(ws)
        m_comb, sd_comb = float((ms * ws).sum() / ws.sum()), float(1.0 / np.sqrt(ws.sum()))
        upper = abs(m_comb) + DEVS * sd_comb
        if upper > worst:
            worst, worst_m, worst_z, worst_at = upper, m_comb, m_comb / sd_comb, (pa, pc)
    pattern = None if worst_at is None else ("=" if worst_at[0] else "!=") + "/" + ("=" if worst_at[1] else "!=")
    return dict(upper=worst, m=worst_m, z=worst_z, values=len(present), pattern=pattern)


def mediation_test(s: PairSample, a: str, b: str, c: str) -> dict:
    """Does B explain the A-C link? (equivalence test, see the module docstring)"""
    link = s.link(a, c)
    bound = (1 - EXPLAINED) * link
    e = effect_given(s, a, b, c)
    return dict(a=a, b=b, c=c, link=round(link, 4), bound=round(bound, 4), m_given_b=round(e["m"], 4),
                upper=round(e["upper"], 4), share_upper=round(e["upper"] / link, 3) if link > 0 else float("inf"),
                worst_pattern=e["pattern"], values_of_b=e["values"], explained=bool(e["upper"] < bound))


def find_splits(graph: nx.Graph, s: PairSample):
    """Test every triangle, then apply the splits from the best explained up while the triangle
    still exists. Returns (graph without the split edges, splits applied, every test)."""
    tests = [mediation_test(s, a, b, c) for a, c in graph.edges for b in sorted(nx.common_neighbors(graph, a, c))]
    g = graph.copy()
    splits = []
    for t in sorted((t for t in tests if t["explained"]), key=lambda t: t["share_upper"]):
        a, b, c = t["a"], t["b"], t["c"]
        if g.has_edge(a, c) and g.has_edge(a, b) and g.has_edge(b, c):
            g.remove_edge(a, c)
            splits.append(t)
    return g, splits, tests


def split_groups(graph: nx.Graph, separated: Set[frozenset]) -> List[List[str]]:
    """Groups: the largest sets of columns that are connected through the edges of `graph` (the
    graph after removing split edges) and contain no separated pair. Computed as: per connected
    component, the largest sets with no separated pair (maximal cliques of "may be together"),
    each cut into its connected pieces, keeping the pieces not contained in a larger one."""
    pieces = set()
    for component in nx.connected_components(graph):
        compatible = nx.Graph()
        compatible.add_nodes_from(component)
        compatible.add_edges_from((x, y) for x, y in itertools.combinations(sorted(component), 2)
                                  if frozenset((x, y)) not in separated)
        for clique in nx.find_cliques(compatible):
            pieces.update(frozenset(p) for p in nx.connected_components(graph.subgraph(clique)))
    maximal = [p for p in pieces if not any(p < q for q in pieces)]
    return sorted((sorted(p) for p in maximal), key=lambda g: (-len(g), g))


def target_dataset(groups: List[List[str]], target: str) -> List[str]:
    """MN: the union of the groups containing the target."""
    hit = [g for g in groups if target in g]
    return sorted(set().union(*hit)) if hit else [target]


def run_v2(lima, graph: nx.Graph, n_pairs: int = N_PAIRS) -> dict:
    """V2's extra step on top of a V1 run (its LIMA object and DIADA graph)."""
    start = time.perf_counter()
    s = PairSample(lima, n_pairs)
    g, splits, tests = find_splits(graph, s)
    groups = split_groups(g, {frozenset((t["a"], t["c"])) for t in splits})
    return dict(graph=g, splits=splits, tests=tests, groups=groups, sample=s, seconds=time.perf_counter() - start)
