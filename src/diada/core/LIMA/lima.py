"""
LIMA, the algorithm behind DIADA's soundness scores: a Python port of the one inside DIADA-0.8.jar
(decompiled with CFR 0.152: edu.upc.Main, lima.Scheduler, utils.BetaDistribution,
data.dataset.CSVDataset). With the same input, approx and seed (42, the jar's) it gives the jar's
output exactly: every cell of the output matrix, on the 4 old benchmarks and the UN and MN versions
of the 17 noise benchmarks (explorations/jar_like_vs_jar.py), and in less time than the jar.

The algorithm, step by step as in the jar:

  1. Columns and predicates. A column whose name contains "Double" is REAL, "Integer" is INTEGER,
     anything else STRING (with DIADA's bucketing every column is STRING). Per column, the
     predicates comparing two rows on it: STRING -> EQ, NE; numeric -> EQ, NE, GT, LE, LT, GE.
  2. Single predicates. Pairs of rows (x, y), x != y, are drawn in batches of 100 * 2^b pairs
     (b capped at 20) with Java's SplittableRandom, until 0.0325 * rows^2 pairs or until every
     predicate is done. A predicate counts batches while grad = b / (a (a+b)^2) of its Beta is
     above minGrad = approx * 0.01, and keeps the pairs it matched. NE, GT, LT of a constant column
     are never counted (Beta(1, 1e9 + 1)).
  3. Patterns. The predicates are sorted by mean log-odds, most frequent first. For each predicate
     (the root of a path), depth first: add a more frequent predicate of another column, counting
     the bigger pattern by filtering the pairs stored in the smaller one (no new sampling). A
     pattern is explored only if every sub-pattern edge it builds on was sound and non-empty.
  4. The test. Edge X -> X + p: Beta(1 + count(X + p), 1 + count(X) - count(X + p)). For each q in
     X, against the edge X - q -> X - q + p:
        normDist = (meanLogOdds(lower) - meanLogOdds(upper)) / sqrt(var(lower) + var(upper))
     The edge is sound (can be extended) if every normDist >= 2. Every edge is reported with its
     smallest normDist; the output for (column f, column t) is the largest over the edges that add
     t to a pattern containing f. DIADA then keeps soundness > 4.

Two differences with the jar, neither changing an edge in the checks above:
  - When an edge fails, the jar stops at the first comparison below 2 (in its hash-map order) and
    reports the smallest value so far; here it is the smallest over all comparisons. Both are below
    2, so no edge above DIADA's threshold changes.
  - The jar identifies a pattern by an incremental 32-bit hash (lattice.PredSet: add computes
    (hash + (p+1) * predProd[col]) % MOD, remove computes (hash - score + MOD) % MOD). The int
    arithmetic overflows, and a pattern reached by removing a predicate can get a different hash
    than when it was built by adding (e.g. 2 vs -333333311). The lookup then misses, the jar makes
    an empty copy of the pattern, and exploreNode prunes the branch. This port does not copy the
    bug. On DIADA's bucketed input (only EQ/NE) the bug never changed a value in the 38 runs. With
    numeric columns (num_buckets <= 1) the port explores a few more patterns: on bank, students,
    pendigits and diamonds, 0-20 output values differ and every edge is the same.

Variants change the test by overriding LIMA.judge (see explorations/v1_two_sided.py).
"""

from __future__ import annotations

import io
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .java_random import JavaSplittableRandom

DEFAULT_APPROX = 1e-6  # what DIADA has always passed to the jar
DEFAULT_SEED = 42  # the jar's default seed

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))


# ---------------------------------------------------------------- utils.BetaDistribution

def _tables():
    y1 = [-0.5772156649015329]
    for s in range(1, 1000):
        y1.append(y1[-1] + float(np.float32(1.0) / np.float32(s)))
    y2 = [math.pi ** 2 / 6.0]
    for s in range(1, 500):
        v = 1.0 / s
        y2.append(y2[-1] - v * v)
    return y1, y2


_Y1, _Y2 = _tables()


def y1(n: int) -> float:
    """the jar's digamma(n): -gamma + harmonic numbers (float steps), log(n) from 1000"""
    return math.log(n) if n >= 1000 else _Y1[n - 1]


def y2(n: int) -> float:
    """the jar's trigamma(n): pi^2/6 - sum of 1/s^2, 1/n (float) from 500"""
    return float(np.float32(1.0) / np.float32(n)) if n >= 500 else _Y2[n - 1]


class Beta:
    """Beta(a, b) with the jar's statistics: mean log-odds `mlo` and its variance `var` (the jar's
    sdLogOdds, which is a variance)"""
    __slots__ = ("a", "b", "mlo", "var")

    def __init__(self, a: int = 1, b: int = 1):
        self.a, self.b = a, b
        self.mlo = y1(a) - y1(b)
        self.var = y2(a) + y2(b)

    @property
    def grad(self) -> float:
        s = float(self.a + self.b)
        return self.b / (self.a * s * s)


def norm_dist(lower: Beta, upper: Beta) -> float:
    return (lower.mlo - upper.mlo) / math.sqrt(lower.var + upper.var)


# ---------------------------------------------------------------- data and predicates

EQ, NE, GT, LE, LT, GE = range(6)


def column_kind(name: str) -> str:
    return "REAL" if "Double" in name else "INTEGER" if "Integer" in name else "STRING"


class Dataset:
    """data.dataset.CSVDataset: the columns, their kinds (from the names), their values (STRING as
    codes) and which are constant."""

    def __init__(self, columns: List[str], values: List[np.ndarray], size: Optional[int] = None):
        self.columns = list(columns)
        self._size = len(values[0]) if values else int(size or 0)
        self.kinds = [column_kind(c) for c in self.columns]
        self.values = values
        self.const = [bool(np.all(v == v[0])) for v in values]  # NaN != NaN: never constant, as in Java

    @property
    def size(self) -> int:
        return self._size

    @classmethod
    def from_csv(cls, source, nrow: Optional[int] = None) -> "Dataset":
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, nrows=nrow)
        values = []
        for c in raw.columns:
            kind = column_kind(c)
            if kind == "REAL":
                values.append(np.array([float(s) if s else np.nan for s in raw[c]], dtype=np.float32))
            elif kind == "INTEGER":
                values.append(raw[c].astype(np.int64).to_numpy())
            else:
                values.append(pd.factorize(raw[c])[0].astype(np.int32))
        return cls(list(raw.columns), values)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Dataset":
        """the values as the jar would read them from the CSV DIADA writes"""
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return cls.from_csv(buf)

    def expand(self, n: int, m: int) -> "Dataset":
        """n rows and m columns drawn with repetition (np.random), so a bigger dataset can be made
        from a smaller one; each column keeps its name (so its kind) plus "_<position>"."""
        rows = np.random.randint(0, self.size, size=n)
        cols = np.random.randint(0, len(self.columns), size=m)
        return Dataset([f"{self.columns[c]}_{i}" for i, c in enumerate(cols)], [self.values[c][rows] for c in cols], n)

    def predicates(self) -> List[Tuple[int, int]]:
        """every (column, operator), in the jar's order"""
        return [(c, p) for c, k in enumerate(self.kinds) for p in range(2 if k == "STRING" else 6)]


def holds(values: np.ndarray, op: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """which pairs (xs[i], ys[i]) satisfy `values[x] op values[y]`"""
    a, b = values[xs], values[ys]
    if op == EQ:
        return a == b
    if op == NE:
        return a != b
    if op == GT:
        return a > b
    if op == LE:
        return a <= b
    if op == LT:
        return a < b
    return a >= b


# ---------------------------------------------------------------- lima.Scheduler

class LIMA:
    """lima.Scheduler.populatePredicates (with ar = true). After run():
         matrix  pairs[f][t], as the jar prints it
         node    pattern (frozenset of (column, operator)) -> Beta of its share of the pairs
         edge    (pattern, (column, operator)) -> [Beta, sound]
         stats   rows, columns, predicates, batches and pairs drawn, patterns explored"""

    explore = 2.0  # an edge is extended only if every comparison reaches this

    def __init__(self, dataset: Dataset, approx: float = DEFAULT_APPROX, seed: int = DEFAULT_SEED):
        self.dataset = dataset
        self.min_grad = float(np.float32(approx)) * 0.01  # Float.parseFloat(args[1]) * 0.01
        self.seed = seed
        self.preds = dataset.predicates()
        self.node, self.edge, self.tps = {}, {}, {}
        n_cols = len(dataset.columns)
        self.matrix = np.zeros((n_cols, n_cols))
        self.stats = dict(rows=dataset.size, columns=n_cols, predicates=len(self.preds), batches=0, pairs_sampled=0,
                          patterns=0)

    def judge(self, distances: List[float]) -> Tuple[float, bool]:
        """(the edge's score, whether it is sound) from its comparisons' normDist"""
        return min(distances), all(d >= self.explore for d in distances)

    def run(self) -> np.ndarray:
        self.count_predicates()
        order = sorted(range(len(self.preds)), key=lambda i: -self.node[frozenset([self.preds[i]])].mlo)
        self.order = [self.preds[i] for i in order]
        for i in range(len(self.order)):
            self.search(frozenset([self.order[i]]), i)
        return self.matrix

    def count_predicates(self) -> None:
        """step 2: the single predicates, on batches of pairs, keeping the pairs each one matched"""
        data, size = self.dataset.values, self.dataset.size
        rng = JavaSplittableRandom(self.seed)
        active, counts, kept = [], {}, {}
        for cp, p in self.preds:
            key = frozenset([(cp, p)])
            self.node[key] = Beta()
            self.edge[(frozenset(), (cp, p))] = [Beta(), True]
            kept[key], counts[key] = ([], []), [0, 0, 0]  # matched, not matched, pairs in its batches
            if self.dataset.const[cp] and p in (NE, GT, LT):
                self.node[key] = Beta(1, 1 + 1_000_000_000)
                continue
            active.append((cp, p))
        nn, total, batches = size * size * 0.0325, 0, 0
        while total < nn:
            total += 100 * (1 << min(batches, 20))
            batches += 1
        for batch in range(batches):
            n_batch = 100 * (1 << min(batch, 20))
            hits = dict.fromkeys(active, 0)
            for start in range(0, n_batch, 1 << 21):  # the batch in pieces, in the jar's order
                x, y = rng.pairs(min(1 << 21, n_batch - start), size)
                for cp, p in active:
                    m = holds(data[cp], p, x, y)
                    xs, ys = kept[frozenset([(cp, p)])]
                    xs.append(x[m])
                    ys.append(y[m])
                    hits[(cp, p)] += int(m.sum())
            self.stats["batches"] += 1
            self.stats["pairs_sampled"] += n_batch
            keep = []
            for cp, p in active:
                key = frozenset([(cp, p)])
                c = counts[key]
                c[0] += hits[(cp, p)]
                c[1] += n_batch - hits[(cp, p)]
                c[2] += n_batch
                self.node[key] = Beta(1 + c[0], 1 + c[1])
                self.edge[(frozenset(), (cp, p))][0] = Beta(1 + c[0], 1 + c[1])  # the root edge: the same counts
                if self.node[key].grad > self.min_grad:
                    keep.append((cp, p))
            active = keep
            if not active:
                break
        for key, (xs, ys) in kept.items():
            self.tps[key] = (np.concatenate(xs) if xs else np.empty(0, np.int32),
                             np.concatenate(ys) if ys else np.empty(0, np.int32), counts[key][2])

    def explorable(self, pattern: frozenset, new) -> bool:
        """Scheduler.exploreNode"""
        if self.node[pattern].a == 1:
            return False
        for q in pattern:
            e = self.edge.get((pattern - {q}, new))
            if e is None or not e[1] or self.node[(pattern - {q}) | {new}].a == 1:
                return False
        return True

    def propagate(self, pattern: frozenset, new) -> frozenset:
        """Scheduler.propagateAcross: count pattern + new on the pattern's pairs, test the edge"""
        xs, ys, n = self.tps[pattern]
        m = holds(self.dataset.values[new[0]], new[1], xs, ys)
        child = pattern | {new}
        self.tps[child] = (xs[m], ys[m], n)
        a = int(m.sum())
        self.node[child] = Beta(1 + a, 1 + n - a)
        upper = Beta(1 + a, 1 + len(xs) - a)
        score, sound = self.judge([norm_dist(self.edge[(pattern - {q}, new)][0], upper) for q in pattern])
        self.edge[(pattern, new)] = [upper, sound]
        t = new[0]
        for f, _ in pattern:
            if score > self.matrix[f, t]:
                self.matrix[f, t] = score
        self.stats["patterns"] += 1
        return child

    def search(self, pattern: frozenset, last: int) -> None:
        """Scheduler.search: add the more frequent predicates, depth first, freeing pairs on the way back"""
        used = {cp for cp, _ in pattern}
        for k in range(last):
            new = self.order[k]
            if new[0] in used or not self.explorable(pattern, new):
                continue
            child = self.propagate(pattern, new)
            self.search(child, k)
            del self.tps[child]


def run_lima(source, nrow: Optional[int] = None, approx: float = DEFAULT_APPROX, seed: int = DEFAULT_SEED,
             engine: type = LIMA) -> Tuple[np.ndarray, List[str]]:
    """LIMA on a CSV (path or buffer) or a DataFrame: (pairs[f][t] matrix, column names)"""
    dataset = Dataset.from_frame(source) if isinstance(source, pd.DataFrame) else Dataset.from_csv(source, nrow)
    return engine(dataset, approx=approx, seed=seed).run(), dataset.columns
