"""
V2.exact: V2 (two-sided test + splitting a triangle A-B-C into {A, B} and {B, C} when B explains
A-C) with exact counts over ALL row pairs instead of LIMA's sample of pairs.

LIMA's predicates are kept: two rows "agree" on a column when they have the same value
(DIADA's 10-bucket labelling). For two or three columns the number of agreeing pairs needs no
sampling: a group of n rows with the same values gives n * (n - 1) ordered pairs that agree, so
one pass over the rows gives the exact count over all N * (N - 1) pairs.

Edges. For every pair of columns x, y:
    phi(x, y)   the correlation between the predicates "agree on x" and "agree on y" over all
                pairs (it sums up LIMA's four agree/differ patterns; positive = rows that agree
                on x agree more often on y)
    z_row       how far phi is from chance, in standard deviations of phi over K shuffles of
                y's rows (a shuffle keeps each column's values and breaks any link). The
                uncertainty comes from the rows: with all pairs counted, a z over pairs would
                call almost anything sound (pure-noise columns of bank_mn get z = 25-70).
    edge if |z_row| > 4 (DIADA's threshold), in either direction, AND |phi| >= 0.01. With all
    pairs counted, almost every real relation beats chance on tens of thousands of rows (bank:
    102 of 136 column pairs), so a minimum strength is needed; 0.01 keeps 28 of the jar's 29
    bank edges and 21 of V1's 22. Pairs below it are not shuffled at all.

Triangles. For every triangle A-B, B-C, A-C and each value v of B:
    phi_v(A, C) over the pairs whose two rows both have B = v
    phi(A, C | B) = |phi_v| averaged over v, weighted by each value's share of those pairs,
                minus its average over 5 shuffles of A within each value of B (what chance
                alone gives; |phi_v| is never 0 in finite data)
    left = phi(A, C | B) / |phi(A, C)|: the share of the A-C link left once B is known
    split (A-C explained by B) if left <= 0.2 for some B; otherwise keep the triangle, also
    when unsure (true mediation gives 0.03-0.10 on the synthetic benchmarks).

Grouping: as V2 (explorations/split_grouping.py): split triangles are cut at B, everything
connected without a split stays together.

The soundness table it returns holds |z_row| per pair (0 below the strength floor; both
directions), so DIADA's own build_soundness_graph with threshold 4 gives the same graph.
"""

from __future__ import annotations

import itertools
import time

import networkx as nx
import numpy as np
import pandas as pd

from diada.core.diada_tool import _label_columns

from split_grouping import split_groups

K_SHUFFLES = 20  # per pair: the spread of phi under chance
K_TRIANGLE = 5  # per triangle: only the average under chance is needed
PHI_MIN = 0.01
SPLIT_LEFT = 0.2
DEVS = 4.0


def bucket_codes(df: pd.DataFrame) -> dict:
    """Column -> integer codes after DIADA's labelling with 10 buckets (what LIMA compares)."""
    lab = _label_columns(df.copy(), 10)
    return {c: pd.factorize(lab.iloc[:, i].astype(str))[0].astype(np.int64) for i, c in enumerate(df.columns)}


def _same(counts: np.ndarray) -> np.ndarray:
    """Ordered agreeing pairs from group sizes, summed over the last axis."""
    counts = counts.astype(np.float64)
    return (counts * (counts - 1)).sum(-1)


def _phi(s_xy, s_x, s_y, total):
    """Agreement correlation from pair counts (arrays broadcast)."""
    p_x, p_y, p_xy = s_x / total, s_y / total, s_xy / total
    den = np.sqrt(p_x * (1 - p_x) * p_y * (1 - p_y))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (p_xy - p_x * p_y) / den, np.nan)


def pair_tests(codes: dict, rng: np.random.Generator, k: int = K_SHUFFLES) -> pd.DataFrame:
    """phi and z_row for every pair of columns (shuffles shared across pairs)."""
    cols = list(codes)
    N = len(codes[cols[0]])
    total = float(N) * (N - 1)
    card = {c: int(codes[c].max()) + 1 for c in cols}
    s1 = {c: _same(np.bincount(codes[c], minlength=card[c])) for c in cols}
    perms = np.stack([rng.permutation(N) for _ in range(k)])
    rows = []
    for x, y in itertools.combinations(cols, 2):
        nx_, ny_ = card[x], card[y]
        s_xy = _same(np.bincount(codes[x] * ny_ + codes[y], minlength=nx_ * ny_))
        f = float(_phi(s_xy, s1[x], s1[y], total))
        if not abs(f) >= PHI_MIN:  # too weak to be an edge: no need to measure chance
            rows.append(dict(column_1=x, column_2=y, phi=f, z_row=np.nan))
            continue
        shuffled_y = codes[y][perms]  # k x N
        idx = (np.arange(k)[:, None] * (nx_ * ny_) + codes[x][None, :] * ny_ + shuffled_y).ravel()
        s_null = _same(np.bincount(idx, minlength=k * nx_ * ny_).reshape(k, nx_ * ny_))
        null = _phi(s_null, s1[x], s1[y], total)
        sd = float(np.nanstd(null))
        z = (f - float(np.nanmean(null))) / sd if sd > 0 else (np.inf if abs(f) > 0 else 0.0)
        rows.append(dict(column_1=x, column_2=y, phi=f, z_row=float(z) if np.isfinite(f) else 0.0))
    return pd.DataFrame(rows)


def _conditional_phi(a, b, c, na, nb, nc) -> np.ndarray:
    """phi(A, C | B) (weighted mean of |phi_v|) for a batch of A columns: a is (k, N)."""
    k = a.shape[0]
    n_b = np.bincount(b, minlength=nb).astype(np.float64)
    t_b = n_b * (n_b - 1)
    s_c = _same(np.bincount(b * nc + c, minlength=nb * nc).reshape(nb, nc))
    off = np.arange(k)[:, None]
    s_a = _same(np.bincount((off * nb * na + b[None, :] * na + a).ravel(), minlength=k * nb * na).reshape(k, nb, na))
    s_ac = _same(np.bincount((off * nb * na * nc + (b[None, :] * na + a) * nc + c[None, :]).ravel(),
                             minlength=k * nb * na * nc).reshape(k, nb, na * nc))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = _phi(s_ac, s_a, s_c[None, :], np.where(t_b > 0, t_b, np.nan)[None, :])
    w = t_b / t_b.sum()
    return np.nansum(np.abs(f) * w[None, :], axis=1)


class _WithinShuffles:
    """K permutations of the rows within each value of a column, cached per column."""

    def __init__(self, codes: dict, rng: np.random.Generator, k: int):
        self.codes, self.rng, self.k, self.cache = codes, rng, k, {}

    def __call__(self, b: str) -> np.ndarray:
        if b not in self.cache:
            v = self.codes[b]
            order = np.argsort(v, kind="stable")
            perms = np.empty((self.k, len(v)), dtype=np.int64)
            for i in range(self.k):
                p = np.lexsort((self.rng.random(len(v)), v))  # sorted by b, random within
                perms[i, order] = p
            self.cache[b] = perms
        return self.cache[b]


def triangle_tests(codes: dict, graph: nx.Graph, phis: dict, rng: np.random.Generator, k: int = K_TRIANGLE) -> list:
    """For every triangle edge a-c and common neighbour b: phi(a, c | b) above chance and the
    share of the a-c link it leaves."""
    within = _WithinShuffles(codes, rng, k)
    card = {c: int(codes[c].max()) + 1 for c in codes}
    out = []
    for a, c in graph.edges:
        for b in sorted(nx.common_neighbors(graph, a, c)):
            obs = _conditional_phi(codes[a][None, :], codes[b], codes[c], card[a], card[b], card[c])[0]
            null = _conditional_phi(codes[a][within(b)], codes[b], codes[c], card[a], card[b], card[c])
            above = max(0.0, obs - float(null.mean()))
            left = above / abs(phis[frozenset((a, c))])
            out.append(dict(a=a, c=c, mediator=b, phi_ac=round(float(phis[frozenset((a, c))]), 5),
                            phi_given_b=round(float(obs), 5), chance=round(float(null.mean()), 5), left=round(float(left), 3)))
    return out


def run_exact(df: pd.DataFrame, seed: int = 0, k: int = K_SHUFFLES) -> dict:
    """V2.exact on df. Returns the soundness table (|z_row| per pair, both directions), the
    split groups, the triangles split and tested, and the time of each step."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    codes = bucket_codes(df)
    t1 = time.perf_counter()
    pairs = pair_tests(codes, rng, k)
    t2 = time.perf_counter()
    edges = pairs[(pairs.z_row.abs() > DEVS) & (pairs.phi.abs() >= PHI_MIN)]
    graph = nx.Graph()
    graph.add_edges_from(zip(edges.column_1, edges.column_2))
    phis = {frozenset((x, y)): f for x, y, f in zip(pairs.column_1, pairs.column_2, pairs.phi)}
    tested = triangle_tests(codes, graph, phis, rng)
    pruned, g = [], graph.copy()
    for t in sorted(tested, key=lambda t: t["left"]):
        if t["left"] > SPLIT_LEFT:
            break
        a, b, c = t["a"], t["mediator"], t["c"]
        if g.has_edge(a, c) and g.has_edge(a, b) and g.has_edge(b, c):
            g.remove_edge(a, c)
            pruned.append(t)
    groups = split_groups(graph, {frozenset((t["a"], t["c"])) for t in pruned})
    t3 = time.perf_counter()
    gone = {frozenset((t["a"], t["c"])) for t in pruned}
    score = np.where(pairs.phi.abs() >= PHI_MIN, pairs.z_row.abs().fillna(0.0), 0.0)
    sound = pd.concat([pairs.assign(soundness=score),
                       pairs.rename(columns={"column_1": "column_2", "column_2": "column_1"}).assign(soundness=score)])
    keep = np.array([frozenset((x, y)) not in gone for x, y in zip(sound.column_1, sound.column_2)], dtype=bool)
    return dict(lima=None, soundness=sound.loc[keep, ["column_1", "column_2", "soundness"]].reset_index(drop=True),
                soundness_unpruned=sound[["column_1", "column_2", "soundness"]].reset_index(drop=True),
                pair_table=pairs, pairs=len(df) * (len(df) - 1), groups=groups, pruned=pruned, tested=tested,
                rounds=None, nodes=None,
                seconds=t3 - t0, seconds_bucketing=t1 - t0, seconds_pairs=t2 - t1, seconds_prune=t3 - t2)
