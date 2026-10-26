"""
Point 2 of the methodology review: can agree/differ predicates miss a dependence the values show?

For two categorical columns with joint shares p(x, y) and e(x, y) = p(x, y) - p(x) p(y):
    distance from independence    D   = sum e^2                      (0 only if independent)
    what pairs measure            cov = Cov(agree on X, agree on Y)  = sum e^2 + 2 sum p(x) p(y) e
    the part pairs miss           2 sum p(x) p(y) e                   (can cancel D)
    triple-corrected statistic    HSIC = cov - 2 [P(i~j on X, i~k on Y) - P(agree X) P(agree Y)] = sum e^2
where "i~k on Y" compares row i with a THIRD row k: the correction needs triples of rows, still
only agree/differ predicates.

Part 1, real data: for every column pair of diamonds, bank, house_sales (10-bucket data), and for
every value v of B in the triangles we have studied, the retention r = cov / D: 1 = the pair
statistic sees the full distance, < 1 diluted, <= 0 blind or inverted.
Part 2, a constructed triangle where B does NOT explain A-C (the values stay dependent within each
value of B) but the pair statistic is exactly 0 within each value: V2's predicate test would split
it; the value-level measure and the triple-corrected statistic would not.

Run with:  python explorations/blind_spot_demo.py
Writes results/blind_spot/{pairs.csv, strata.csv, constructed.txt}
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402

OUT = paths.RESULTS_DIR / "blind_spot"


def parts(P: np.ndarray) -> dict:
    """For a joint table P (shares): D, cov, the missed part, and value-level MI (bits)."""
    px, py = P.sum(1), P.sum(0)
    E = P - np.outer(px, py)
    D = float((E ** 2).sum())
    cov = float((P ** 2).sum() - (px ** 2).sum() * (py ** 2).sum())
    nz = P > 0
    mi = float((P[nz] * np.log2(P[nz] / np.outer(px, py)[nz])).sum())
    return dict(D=D, cov=cov, missed=cov - D, retention=cov / D if D > 0 else np.nan, mi_bits=mi)


def table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = pd.crosstab(x, y).to_numpy().astype(float)
    return t / t.sum()


def real_pairs() -> pd.DataFrame:
    rows = []
    for name, rel in [("diamonds", "diamonds.csv"), ("bank", "bank.csv"), ("house_sales", "house_sales.csv")]:
        codes = bucket_codes(pd.read_csv(paths.DATA_DIR / "base_datasets" / rel))
        for x, y in itertools.combinations(codes, 2):
            r = parts(table(codes[x], codes[y]))
            rows.append(dict(dataset=name, x=x, y=y, **r))
    return pd.DataFrame(rows)


def strata() -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).parent))
    from equivalence_demo import CASES
    rows = []
    for tri, path, a, b, c in CASES:
        codes = bucket_codes(pd.read_csv(path))
        A, B, C = codes[a], codes[b], codes[c]
        for v in np.unique(B):
            m = B == v
            if m.sum() < 50 or len(np.unique(A[m])) < 2 or len(np.unique(C[m])) < 2:
                continue
            rows.append(dict(triangle=tri, value_of_B=int(v), rows=int(m.sum()), **parts(table(A[m], C[m]))))
    return pd.DataFrame(rows)


def blind_table(rng) -> np.ndarray:
    """A 3x3 joint table with zero pair covariance (agreement on X says nothing about agreement on Y)
    and as much value-level dependence as a random search finds. Along the line P(t) = Q + t E between
    the independent table Q (same margins) and a random table, cov(t) = t^2 sum E^2 + 2 t sum Q E, which
    is 0 again at t* = -2 sum Q E / sum E^2 when sum Q E < 0; keep t* if the table stays valid."""
    best, best_mi = None, -1.0
    for _ in range(100000):
        R = rng.dirichlet(np.full(9, 0.5)).reshape(3, 3)
        Q = np.outer(R.sum(1), R.sum(0))
        E = R - Q
        b = 2 * (Q * E).sum()
        if b >= 0:
            continue
        t = -b / (E ** 2).sum()
        P = Q + t * E
        if (P < 0).any():
            continue
        mi = parts(P)["mi_bits"]
        if mi > best_mi:
            best, best_mi = P, mi
    return best


def constructed(rng) -> str:
    """Triangle A-B-C: B has 2 values; within each value, (A, C) follow the blind table, with A's and
    C's labels shifted by B so that A and C both depend on B."""
    P = blind_table(rng)
    n = 40000
    flat = rng.choice(9, size=n, p=P.ravel())
    a0, c0 = flat // 3, flat % 3
    b = rng.integers(0, 2, n)
    a = np.where(b == 0, a0, (a0 + 1) % 3)     # relabel A when B = 1
    c = np.where(b == 0, c0, (c0 + 2) % 3)     # relabel C when B = 1
    lines = ["Blind table P(A, C) used within each value of B:", np.array2string(P, precision=4)]
    whole = parts(table(a, c))
    lines.append(f"overall A-C:  value MI {whole['mi_bits']:.4f} bits | pair cov {whole['cov']:.5f} | D = HSIC {whole['D']:.5f}")
    val_num = pred_num = hsic_num = 0.0
    for v in (0, 1):
        m = b == v
        s = parts(table(a[m], c[m]))
        w = m.mean()
        val_num += w * s["mi_bits"]
        pred_num += w * s["cov"]
        hsic_num += w * s["D"]
        lines.append(f"within B = {v}: value MI {s['mi_bits']:.4f} bits | pair cov {s['cov']:.6f} | D = HSIC {s['D']:.5f}")
    lines.append(f"share left given B:  values {val_num / whole['mi_bits']:.2f} | pair statistic {pred_num / whole['cov']:.3f} | "
                 f"triple-corrected (HSIC) {hsic_num / whole['D']:.2f}")
    lines.append("-> the pair statistic says B explains A-C (share ~0): V2's predicate test would split; "
                 "values and HSIC say most of the link is left: keep")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = real_pairs()
    pairs.to_csv(OUT / "pairs.csv", index=False)
    dep = pairs[pairs.mi_bits > 0.01]  # pairs with a non-negligible value-level link
    print("Part 1a, column pairs with value-level MI > 0.01 bits: retention r = cov / D")
    print(dep.groupby("dataset").retention.describe(percentiles=[0.1, 0.25, 0.5]).round(2).to_string())
    print("near-blind |r| < 0.1:", round((dep.retention.abs() < 0.1).mean(), 3), "| |r| < 0.25:", round((dep.retention.abs() < 0.25).mean(), 3),
          "| sign flipped (r < 0, still detected two-sided):", round((dep.retention < -0.1).mean(), 3))
    worst = dep.assign(abs_r=dep.retention.abs()).nsmallest(6, "abs_r")[["dataset", "x", "y", "mi_bits", "retention"]]
    print(worst.round(4).to_string(index=False))
    st = strata()
    st.to_csv(OUT / "strata.csv", index=False)
    print("\nPart 1b, per value of B in the studied triangles (value MI > 0.005 bits): retention")
    s2 = st[st.mi_bits > 0.005]
    print(s2.assign(near_blind=s2.retention.abs() < 0.1, flipped=s2.retention < -0.1).groupby("triangle")
          .agg(values=("retention", "count"), near_blind=("near_blind", "sum"), flipped=("flipped", "sum"),
               median_r=("retention", "median")).round(2).to_string())
    text = constructed(np.random.default_rng(0))
    (OUT / "constructed.txt").write_text(text)
    print("\nPart 2, constructed triangle\n" + text)


if __name__ == "__main__":
    main()
