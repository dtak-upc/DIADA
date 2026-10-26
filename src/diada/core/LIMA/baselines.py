"""
The Apriori and FP-Growth baselines of Experiment 1 (the paper's Figure 4), on the same data,
predicates, pair sampler (Java's SplittableRandom) and statistics as LIMA (lima.py).

Unlike LIMA, which decides how many pairs each pattern needs and which patterns to explore, a
baseline draws ONE sample of n pairs (the paper: n = k / eps, k = 5) and counts EVERY pattern of up
to `depth` predicates (on distinct columns) on it, then scores every edge with LIMA's test:
    edge X -> X + p:  Beta(1 + count(X + p), 1 + count(X) - count(X + p)),  count(empty) = n
    score            the smallest normDist against the edges X - q -> X - q + p
    matrix[f][t]     the largest score over the edges adding t to a pattern containing f

  Apriori    every predicate's matching pairs computed once; patterns built depth first, most
             frequent predicates first, by filtering the parent's pairs
  FP-Growth  pairs that behave the same on every predicate are merged first (with a weight);
             patterns are built depth first, predicates in their default order, by filtering the
             merged pairs
Both give the same counts on the same sample; they differ in time and memory only.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .java_random import JavaSplittableRandom
from .lima import DEFAULT_SEED, EQ, GE, GT, LE, LT, NE, Beta, Dataset, holds, norm_dist


class Exhaustive:
    def __init__(self, dataset: Dataset, n_pairs: int, depth: int = 3, seed: int = DEFAULT_SEED):
        self.dataset, self.n, self.depth, self.seed = dataset, int(n_pairs), depth, seed
        self.preds = dataset.predicates()
        self.counts: Dict[frozenset, int] = {}

    def run(self) -> np.ndarray:
        x, y = JavaSplittableRandom(self.seed).pairs(self.n, self.dataset.size)
        self.count(x, y)
        return self.score()

    def count(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def score(self) -> np.ndarray:
        c, n = self.counts, self.n
        matrix = np.zeros((len(self.dataset.columns),) * 2)
        for pattern, a in c.items():
            if len(pattern) < 2:
                continue
            for new in pattern:
                x = pattern - {new}
                upper = Beta(1 + a, 1 + c[x] - a)
                score = min(norm_dist(Beta(1 + c[x - {q} | {new}], 1 + c.get(x - {q}, n) - c[x - {q} | {new}]), upper)
                            for q in x)
                for f, _ in x:
                    if score > matrix[f, new[0]]:
                        matrix[f, new[0]] = score
        return matrix


class Apriori(Exhaustive):
    def count(self, x: np.ndarray, y: np.ndarray) -> None:
        data = self.dataset.values
        masks = [holds(data[cp], p, x, y) for cp, p in self.preds]
        order = sorted(range(len(self.preds)), key=lambda i: -masks[i].sum())  # most frequent first

        def search(pattern: frozenset, idx: np.ndarray, last: int) -> None:
            self.counts[pattern] = len(idx)
            if len(pattern) >= self.depth:
                return
            used = {cp for cp, _ in pattern}
            for k in range(last):
                i = order[k]
                if self.preds[i][0] in used:
                    continue
                search(pattern | {self.preds[i]}, idx[masks[i][idx]], k)

        everything = np.arange(len(x))
        for k in range(len(order)):
            i = order[k]
            search(frozenset([self.preds[i]]), everything[masks[i]], k)


class FPGrowth(Exhaustive):
    def count(self, x: np.ndarray, y: np.ndarray) -> None:
        # per column, the comparison of the two rows: 0 equal, 1 x < y, 2 x > y, 3 unordered (NaN);
        # it decides every predicate of the column
        codes = np.empty((len(x), len(self.dataset.columns)), dtype=np.int8)
        for c, v in enumerate(self.dataset.values):
            a, b = v[x], v[y]
            codes[:, c] = np.where(a == b, 0, np.where(a < b, 1, np.where(a > b, 2, 3)))
        behaviours, weights = np.unique(codes, axis=0, return_counts=True)
        truth = {EQ: (0,), NE: (1, 2, 3), GT: (2,), LE: (0, 1), LT: (1,), GE: (0, 2)}
        masks = [np.isin(behaviours[:, cp], truth[p]) for cp, p in self.preds]

        def search(pattern: frozenset, idx: np.ndarray, last: int) -> None:
            self.counts[pattern] = int(weights[idx].sum())
            if len(pattern) >= self.depth:
                return
            used = {cp for cp, _ in pattern}
            for k in range(last):
                if self.preds[k][0] in used:
                    continue
                search(pattern | {self.preds[k]}, idx[masks[k][idx]], k)

        everything = np.arange(len(behaviours))
        for k in range(len(self.preds)):
            search(frozenset([self.preds[k]]), everything[masks[k]], k)


BASELINES = {"apriori": Apriori, "FP-GROWTH": FPGrowth}
