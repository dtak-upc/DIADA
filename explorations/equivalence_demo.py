"""
Point 1 of the methodology review: deciding that B explains A-C (A independent of C given B)
needs evidence FOR equality, not only the absence of evidence against it. This script puts the
candidate criteria side by side on triangles whose answer we know or have discussed.

Per triangle (DIADA's 10-bucket data):
  I(A;C), I(A;C|B)   information A and C share, overall and once B is known (bits, Miller-Madow)
  share              I(A;C|B) / I(A;C): how much of the A-C link is left once B is known
  share 90% interval bootstrap over ROWS (200 resamples)
  G-test p           classic test of "A independent of C given B" (H0 = independence)
  BIC                independence preferred if G < df * ln(n)
  equivalence(0.2)   split if the upper end of the interval is below 0.2 (B explains >= 80%),
                     keep if the lower end is above 0.2, otherwise inconclusive (keep)

Run with:  python explorations/equivalence_demo.py
Writes results/equivalence/table.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402

OUT = paths.RESULTS_DIR / "equivalence"
SYN = paths.RESULTS_DIR / "synthetic_triads" / "data"
BASE = paths.DATA_DIR / "base_datasets"
CASES = [
    ("synthetic chain (true mediation)", SYN / "chain.csv", "A", "B", "C"),
    ("synthetic hub_join (true mediation, weak link)", SYN / "hub_join.csv", "A", "B", "C"),
    ("synthetic direct triangle (real A-C link)", SYN / "direct_triangle.csv", "A", "B", "C"),
    ("diamonds: carat-price via y", BASE / "diamonds.csv", "carat", "y", "price"),
    ("house_sales: bathrooms-price via sqft_living", BASE / "house_sales.csv", "bathrooms", "sqft_living", "price"),
    ("bank: previous-Class via poutcome", BASE / "bank.csv", "V15", "V16", "Class"),
    ("bank: pdays-month via housing_loan", BASE / "bank.csv", "V14", "V7", "V11"),
    ("house_sales: price-yr_built via sqft_living", BASE / "house_sales.csv", "price", "sqft_living", "yr_built"),
]
BOOT = 200


def h(*cols, n_cells=None) -> float:
    """Entropy in bits with the Miller-Madow correction."""
    j = cols[0].astype(np.int64)
    for c in cols[1:]:
        j = j * (int(c.max()) + 1) + c
    cnt = np.bincount(j)
    cnt = cnt[cnt > 0].astype(float)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum() + (len(cnt) - 1) / (2 * cnt.sum() * np.log(2)))


def info(a, b, c):
    i_ac = h(a) + h(c) - h(a, c)
    i_ac_b = h(a, b) + h(c, b) - h(a, b, c) - h(b)
    return i_ac, i_ac_b


def gtest(a, b, c):
    """G and df for A independent of C given B (plug-in counts, per value of B)."""
    G, df = 0.0, 0
    for v in np.unique(b):
        m = b == v
        t = pd.crosstab(a[m], c[m]).to_numpy().astype(float)
        if t.shape[0] < 2 or t.shape[1] < 2:
            continue
        e = np.outer(t.sum(1), t.sum(0)) / t.sum()
        nz = t > 0
        G += 2 * (t[nz] * np.log(t[nz] / e[nz])).sum()
        df += (t.shape[0] - 1) * (t.shape[1] - 1)
    return G, df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for name, path, a, b, c in CASES:
        codes = bucket_codes(pd.read_csv(path))
        A, B, C = codes[a], codes[b], codes[c]
        n = len(A)
        i_ac, i_ac_b = info(A, B, C)
        shares = []
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            x, y = info(A[idx], B[idx], C[idx])
            shares.append(y / x if x > 0 else np.nan)
        lo, hi = np.nanquantile(shares, [0.05, 0.95])
        G, df = gtest(A, B, C)
        p = chi2.sf(G, df) if df else 1.0
        share = i_ac_b / i_ac
        verdict = "split (explained)" if hi < 0.2 else "keep (not explained)" if lo > 0.2 else "inconclusive (keep)"
        rows.append(dict(triangle=name, rows=n, I_AC=round(i_ac, 4), I_AC_given_B=round(i_ac_b, 4), share=round(share, 3),
                         share_low=round(lo, 3), share_high=round(hi, 3), G=round(G, 1), df=df, p=p,
                         ttest_says="independent" if p >= 0.01 else "dependent",
                         bic_says="independent" if G < df * np.log(n) else "dependent",
                         equivalence_02=verdict))
        print(name, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
