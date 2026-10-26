"""
Toy example for the union-search discussion: what does a candidate table T add to a query table Q?

Q = flats in one city (zone, size, rooms, year, price). Six candidates with the same schema:
  T1 same     another sample from the same generator (more of the same)
  T2 bigger   bigger flats, same price per m2 rule (new region, same relations)
  T3 pricier  same flats, prices x1.5 per m2 (same relations' shape, different level)
  T4 regulated price set by zone and building year, not by size (a genuinely different relation)
  T5 noise    every column shuffled independently (same values, no relations)
  T6 misalign "rooms" is really the floor number (wrong column match)
  T7 small    same generator as Q but only 300 rows

Two views:
  A. proposed idea: DIADA graph of Q vs DIADA graph of T (jar, buckets=10, edge if soundness > 4)
  B. union view: Q and T stacked with a source column S (Q/T); where does S depend on the columns,
     is that dependence explained by another column (min over y of I(S;x|y), level-1 as in V2/V3), and
     does it appear only once another column is known (max over y of I(S;x|y): x relates to y differently)?
     Exact mutual information in bits on common quantile buckets, with a floor from shuffling S.

Run with:  python extensions/union_search/toy_union.py
Writes extensions/union_search/out/{graphs.csv, union_view.csv}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diada.core.dataset_builder import build_soundness_graph
from diada.core.diada_tool import invoke_diada

OUT = Path(__file__).parent / "out"
N = 3000
ZONES = np.array(["centre", "eixample", "gracia", "sants"])
PPM = {"centre": 6.0, "eixample": 5.0, "gracia": 4.5, "sants": 3.5}  # k EUR per m2


def flats(rng, n=N, size_log_mean=4.3, ppm_factor=1.0):
    zone = rng.choice(ZONES, n)
    size = np.clip(np.exp(rng.normal(size_log_mean, 0.35, n)), 25, 300).round()
    rooms = np.clip(np.round(size / 28 + rng.normal(0, 0.5, n)), 1, 7).astype(int)
    year = rng.integers(1900, 2021, n)
    ppm = np.array([PPM[z] for z in zone]) * ppm_factor
    price = (size * ppm * np.exp(rng.normal(0, 0.15, n))).round()
    return pd.DataFrame({"zone": zone, "size": size, "rooms": rooms, "year": year, "price": price})


def candidates(rng):
    t = {"T1_same": flats(rng), "T2_bigger": flats(rng, size_log_mean=4.8), "T3_pricier": flats(rng, ppm_factor=1.5)}
    reg = flats(rng)
    base = np.array([PPM[z] for z in reg["zone"]]) * 60  # a fixed price per zone ...
    reg["price"] = (base * (1 + (reg["year"] - 1900) / 60) * np.exp(rng.normal(0, 0.1, N))).round()  # ... rising with year
    t["T4_regulated"] = reg
    noise = flats(rng)
    t["T5_noise"] = noise.apply(lambda c: rng.permutation(c.values))
    mis = flats(rng)
    mis["rooms"] = rng.integers(1, 9, N)  # actually the floor number
    t["T6_misalign"] = mis
    t["T7_small"] = flats(rng, n=300)
    return t


def diada_edges(df):
    g = build_soundness_graph(invoke_diada(df, 10, engine="jar", verbose=False), 4)
    return {tuple(sorted(e)) for e in g.edges()}


# ---------- union view: exact MI on common buckets ----------

def buckets(u):
    out = pd.DataFrame(index=u.index)
    for c in u.columns:
        if c == "S" or u[c].dtype == object:
            out[c] = pd.factorize(u[c])[0]
        else:
            out[c] = pd.factorize(pd.qcut(u[c], 10, duplicates="drop"))[0]
    return out


def mi(x, y):
    t = pd.crosstab(x, y).values.astype(float)
    p = t / t.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def cmi(x, y, z):
    return sum((z == v).mean() * mi(x[z == v], y[z == v]) for v in np.unique(z))


def floor(stat, s, rng, strata=None, reps=30):
    """mean + 3 sd of the statistic when S is shuffled (within strata of the conditioning column)."""
    vals = []
    for _ in range(reps):
        if strata is None:
            sp = rng.permutation(s)
        else:
            sp = s.copy()
            for v in np.unique(strata):
                idx = np.where(strata == v)[0]
                sp[idx] = rng.permutation(s[idx])
        vals.append(stat(sp))
    return np.mean(vals) + 3 * np.std(vals)


def union_view(q, t, rng):
    u = pd.concat([q.assign(S="Q"), t.assign(S="T")], ignore_index=True)
    b = buckets(u)
    s = b["S"].values
    rows = []
    cols = [c for c in q.columns]
    for x in cols:
        xv = b[x].values
        m = mi(s, xv)
        m_floor = floor(lambda sp: mi(sp, xv), s, rng)
        # level-1: the column that best explains S-x, and the one that most reveals it
        conds = {y: cmi(s, xv, b[y].values) for y in cols if y != x}
        lo_y, hi_y = min(conds, key=conds.get), max(conds, key=conds.get)
        lo_floor = floor(lambda sp: cmi(sp, xv, b[lo_y].values), s, rng, strata=b[lo_y].values)
        hi_floor = floor(lambda sp: cmi(sp, xv, b[hi_y].values), s, rng, strata=b[hi_y].values)
        shift = m > m_floor
        rows.append({"column": x, "I(S;x)": m, "floor": m_floor, "shift": shift,
                     "min I(S;x|y)": conds[lo_y], "by": lo_y, "direct": shift and conds[lo_y] > lo_floor,
                     "max I(S;x|y)": conds[hi_y], "given": hi_y,
                     "hidden": (not shift) and conds[hi_y] > hi_floor})
    return pd.DataFrame(rows), u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    q = flats(rng)
    cands = candidates(rng)

    gq = diada_edges(q)
    print("Q edges:", sorted(gq))
    graph_rows, union_rows = [], []
    for name, t in cands.items():
        gt = diada_edges(t)
        added, missing = sorted(gt - gq), sorted(gq - gt)
        graph_rows.append({"candidate": name, "edges_T": len(gt), "added": added, "missing": missing,
                           "sym_diff": len(added) + len(missing)})
        uv, u = union_view(q, t, rng)
        s_edges = sorted(e for e in diada_edges(u) if "S" in e)
        uv.insert(0, "candidate", name)
        union_rows.append(uv)
        print(f"\n=== {name}: T edges {len(gt)}, added {added}, missing {missing}")
        print("DIADA edges of S in Q+T:", s_edges)
        print(uv.drop(columns="candidate").round(4).to_string(index=False))

    pd.DataFrame(graph_rows).to_csv(OUT / "graphs.csv", index=False)
    pd.concat(union_rows).to_csv(OUT / "union_view.csv", index=False)


if __name__ == "__main__":
    main()
