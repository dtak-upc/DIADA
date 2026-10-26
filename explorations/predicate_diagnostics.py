"""
Predicate-only diagnostics (no mutual information): exact agreement rates over ALL row pairs
of a dataset, as LIMA's predicates see it (DIADA's 10-bucket labelling, "the two rows agree on
column X"), instead of LIMA's sample of pairs.

For a pair of columns x, y and a pattern (x agree/differ, y agree/differ):
    pairwise effect      m = log P(y pattern | x pattern) - log P(y pattern)
For a triangle a-b-c and a value v of b (pairs whose two rows both have b = v):
    conditional effect   m_v = log P(c pattern | a pattern, b = v) - log P(c pattern | b = v)
m is the quantity whose z LIMA thresholds (nodeDist4 = m / its standard error): m is how much
the agreement rate changes (log scale, 0.1 ~ a 10% change), z is how sure a sample is of it.
expected_z gives the z a sample of n pairs would get if its shares matched the full data.
Because m is on a log scale, a rare pattern (few pairs) can show a huge m; the summary used
for decisions is therefore phi: the correlation between the two predicates "the pair agrees
on x" and "the pair agrees on y" (all four patterns are summed up in it, and rare ones cannot
blow it up; LIMA's z on a 2x2 table is roughly phi times the square root of the pairs).
    phi(x, y)             over all pairs
    phi_v(a, c)           over the pairs whose two rows both have b = v
    phi(a, c | b)         average of |phi_v| over the values v, weighted by their share of the
                          pairs that agree on b

Run with:  python explorations/predicate_diagnostics.py   (prints the synthetic calibration)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diada.core.diada_tool import _label_columns
from diada.core.LIMA.util.stats.betaDist import nodeDist4

sys.path.insert(0, str(Path(__file__).parent))
from lima_mediation import N_PAIRS, _dist  # noqa: E402

PATTERNS = list(itertools.product((True, False), repeat=2))


def bucket_codes(df: pd.DataFrame) -> dict:
    """Column -> integer codes after DIADA's labelling with 10 buckets (what LIMA compares)."""
    lab = _label_columns(df.copy(), 10)
    return {c: pd.factorize(lab.iloc[:, i].astype(str))[0].astype(np.int64) for i, c in enumerate(df.columns)}


def _same(*cols, mask=None) -> float:
    """Number of ordered row pairs (i != j) that agree on all `cols` (within `mask`)."""
    if mask is not None:
        cols = [c[mask] for c in cols]
    if len(cols[0]) < 2:
        return 0.0
    j = cols[0].copy()
    for c in cols[1:]:
        j = j * (int(c.max()) + 1) + c
    n = np.unique(j, return_counts=True)[1].astype(float)
    return float((n * (n - 1)).sum())


def _pattern_counts(x, y, mask=None) -> dict:
    """Ordered pair counts for (x agree?, y agree?) within mask, plus the total."""
    rows = int(mask.sum()) if mask is not None else len(x)
    total = float(rows * (rows - 1))
    xx, yy, xy = _same(x, mask=mask), _same(y, mask=mask), _same(x, y, mask=mask)
    return {(True, True): xy, (True, False): xx - xy, (False, True): yy - xy,
            (False, False): total - xx - yy + xy, "x": {True: xx, False: total - xx},
            "y": {True: yy, False: total - yy}, "total": total}


def effects(counts: dict, all_pairs: float, n: int = N_PAIRS) -> list:
    """Per pattern: (px, py, m, expected z at n sampled pairs, expected pairs in the x pattern)."""
    out = []
    for px, py in PATTERNS:
        both, x_side, y_side, base = counts[(px, py)], counts["x"][px], counts["y"][py], counts["total"]
        if both == 0 or x_side == 0 or y_side == 0:
            m = float("nan")
        else:
            m = np.log(both / x_side) - np.log(y_side / base)
        s = n / all_pairs
        z = nodeDist4(_dist(round(x_side * s), n), _dist(round(both * s), n), _dist(round(base * s), n), _dist(round(y_side * s), n))
        out.append((px, py, m, z, x_side * s))
    return out


def phi(counts: dict) -> float:
    """Correlation between the predicates "agree on x" and "agree on y" over the counted pairs."""
    t = counts["total"]
    if t == 0:
        return float("nan")
    p_xy, p_x, p_y = counts[(True, True)] / t, counts["x"][True] / t, counts["y"][True] / t
    den = np.sqrt(p_x * (1 - p_x) * p_y * (1 - p_y))
    return float((p_xy - p_x * p_y) / den) if den > 0 else float("nan")


def pairwise_phi(codes: dict, x: str, y: str) -> float:
    return phi(_pattern_counts(codes[x], codes[y]))


def conditional_phi(codes: dict, a: str, b: str, c: str) -> dict:
    """phi(a, c | b): |phi_v| averaged over the values v of b (weights: share of the pairs agreeing
    on b), the largest |phi_v| among values holding >= 3% of those pairs, and phi(a, c)."""
    b_agree = _same(codes[b])
    weighted, largest, at = 0.0, 0.0, None
    for v in np.unique(codes[b]):
        mask = codes[b] == v
        cnt = _pattern_counts(codes[a], codes[c], mask)
        f = phi(cnt)
        if cnt["total"] == 0 or f != f:
            continue
        w = cnt["total"] / b_agree
        weighted += w * abs(f)
        if w >= 0.03 and abs(f) > largest:
            largest, at = abs(f), int(v)
    return dict(phi_given_b=round(weighted, 4), largest_phi_v=round(largest, 4), at_value=at,
                phi_overall=round(pairwise_phi(codes, a, c), 4))


def pairwise(codes: dict, x: str, y: str) -> list:
    N = len(codes[x])
    return effects(_pattern_counts(codes[x], codes[y]), N * (N - 1))


def conditional(codes: dict, a: str, b: str, c: str, min_share: float = 0.0) -> pd.DataFrame:
    """Per value v of b: the pairs with b = v (share of all pairs agreeing on b, expected number
    in a sample of 2^17), and per pattern the conditional effect m_v and its expected z."""
    N = len(codes[a])
    all_pairs = N * (N - 1)
    b_agree = _same(codes[b])
    rows = []
    for v in np.unique(codes[b]):
        mask = codes[b] == v
        cnt = _pattern_counts(codes[a], codes[c], mask)
        if cnt["total"] == 0:
            continue
        share = cnt["total"] / b_agree
        if share < min_share:
            continue
        for pa, pc, m, z, n_from in effects(cnt, all_pairs):
            rows.append(dict(value=int(v), share_of_b_pairs=share, pairs_at_2_17=cnt["total"] * N_PAIRS / all_pairs,
                             pa=pa, pc=pc, m=m, expected_z=z))
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame, min_share: float = 0.05) -> dict:
    """Largest |m| and largest |expected z| over the values of b holding at least `min_share` of
    the pairs that agree on b (smaller groups are too thin to say anything at LIMA's sample size)."""
    t = table[table.share_of_b_pairs >= min_share].dropna(subset=["m"])
    if t.empty:
        return dict(max_abs_m=float("nan"), at_value=None, max_abs_z=float("nan"))
    i = t.m.abs().idxmax()
    return dict(max_abs_m=round(abs(t.m[i]), 4), at_value=int(t.value[i]), at_share=round(t.share_of_b_pairs[i], 3),
                max_abs_z=round(t.expected_z.abs().max(), 2))


def main() -> None:
    from diada import paths
    syn = paths.RESULTS_DIR / "synthetic_triads" / "data"
    rows = []
    for name in ["chain", "common_cause", "hub_join", "direct_triangle", "sum_collider", "interaction"]:
        codes = bucket_codes(pd.read_csv(syn / f"{name}.csv"))
        rows.append(dict(benchmark=name, test="A-C given B", **conditional_phi(codes, "A", "B", "C"),
                         phi_AB=round(pairwise_phi(codes, "A", "B"), 4), phi_BC=round(pairwise_phi(codes, "B", "C"), 4)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
