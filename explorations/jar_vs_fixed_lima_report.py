"""
Report for explorations/jar_vs_fixed_lima_diada.py (jar vs. corrected Python LIMA on DIADA's noise
benchmarks), from results/jar_vs_fixed_lima/runs.json.

  selection.csv   per benchmark, version (UN/MN) and Python seed: what DIADA keeps (UN: columns with an
                  edge; MN: the target's group), split into original and injected columns, for the jar
                  and for Python, and how much the two selections overlap
  edge_diffs.csv  every edge found by only one engine (Python: in both seeds = stable, in one = unstable),
                  with the kinds of its two columns, and on the bucketed values: mutual information (bits),
                  the G-test p-value of independence, and each column's number of values
  utility.csv     a quick model (HistGradientBoosting, 5-fold cross-validation on at most 20,000 rows;
                  accuracy or R^2) on: all columns, the original columns only, the jar's selection, and
                  Python's selection (seed 0). A fast proxy, not ARDA.

Run with:  python explorations/jar_vs_fixed_lima_report.py [--no-utility]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402
from jar_vs_fixed_lima_diada import kind  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_fixed_lima"
FILES = {"UN": ("noise_un", "un_noise"), "MN": ("noise_mn", "mn_noise")}


def split(cols, target) -> tuple:
    cols = [c for c in cols if c != target]
    return sum(kind(c) == "original" for c in cols), sum(kind(c) != "original" for c in cols)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 3) if a | b else 1.0


def mi_and_p(x, y) -> tuple:
    t = pd.crosstab(x, y).to_numpy().astype(float)
    n = t.sum()
    e = np.outer(t.sum(1), t.sum(0)) / n
    nz = t > 0
    g = 2 * (t[nz] * np.log(t[nz] / e[nz])).sum()
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return g / (2 * n * np.log(2)), (chi2.sf(g, df) if df else 1.0)


def utility(df: pd.DataFrame, cols, target, task, rng) -> float:
    cols = [c for c in cols if c != target]
    if not cols:
        return float("nan")
    d = df if len(df) <= 20000 else df.sample(20000, random_state=0)
    X = d[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = d[target]
    if task == "classification":
        model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        return float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean())
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
    return float(cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-utility", action="store_true")
    args = parser.parse_args()
    runs = json.loads((OUT / "runs.json").read_text())
    config = yaml.safe_load(open(paths.REPO_ROOT / "benchmarks.yaml"))
    by = {(r["benchmark"], r["variant"], r["engine"], r["seed"]): r for r in runs}
    sel_rows, edge_rows, util_rows = [], [], []
    rng = np.random.default_rng(0)
    for (name, variant, engine, seed), r in sorted(by.items()):
        if engine != "jar" or (name, variant, "lima", 0) not in by:
            continue
        jar = r
        target = jar["target"]
        pys = [by[(name, variant, "lima", s)] for s in (0, 1) if (name, variant, "lima", s) in by]
        for py in pys:
            for what in ("un_kept", "mn_target"):
                jo, ji = split(jar[what], target)
                po, pi = split(py[what], target)
                sel_rows.append(dict(benchmark=name, variant=variant, seed=py["seed"], selection="UN kept" if what == "un_kept" else "MN target group",
                                     jar_original=jo, jar_injected=ji, py_original=po, py_injected=pi,
                                     overlap=jaccard(jar[what], py[what]), jar_s=jar["seconds"], py_s=py["seconds"]))
        df = pd.read_csv(paths.DATA_DIR / FILES[variant][0] / f"{name}_{FILES[variant][1]}.csv")
        codes = bucket_codes(df)
        ej = set(jar["edges"])
        ep = [set(p["edges"]) for p in pys]
        stable, either = set.intersection(*ep), set.union(*ep)
        for e, side in [(e, "jar only") for e in ej - either] + [(e, "python only (both seeds)") for e in stable - ej] + \
                       [(e, "python only (one seed)") for e in (either - stable) - ej]:
            x, y = e.split("|")
            mi, p = mi_and_p(codes[x], codes[y])
            edge_rows.append(dict(benchmark=name, variant=variant, side=side, edge=f"{x}-{y}",
                                  kinds="-".join(sorted((kind(x), kind(y)))), mi_bits=round(mi, 4), p_value=p,
                                  values_x=int(codes[x].max()) + 1, values_y=int(codes[y].max()) + 1))
        if not args.no_utility:
            task = config[name]["task"]
            originals = [c for c in df.columns if kind(c) == "original"]
            for label, cols in (("all columns", list(df.columns)), ("original columns only", originals),
                                ("jar selection", jar["un_kept"] if variant == "UN" else jar["mn_target"]),
                                ("python selection (seed 0)", pys[0]["un_kept"] if variant == "UN" else pys[0]["mn_target"])):
                util_rows.append(dict(benchmark=name, variant=variant, task=task, dataset=label,
                                      columns=len([c for c in cols if c != target]), score=round(utility(df, cols, target, task, rng), 4)))
        print(name, variant, "done", flush=True)
    pd.DataFrame(sel_rows).to_csv(OUT / "selection.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "edge_diffs.csv", index=False)
    if util_rows:
        pd.DataFrame(util_rows).to_csv(OUT / "utility.csv", index=False)


if __name__ == "__main__":
    main()
