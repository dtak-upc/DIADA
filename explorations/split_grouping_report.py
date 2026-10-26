"""
What the split grouping (V3, explorations/split_grouping.py) does to DIADA's MN datasets on
the old benchmarks, from results/jar_vs_two_sided/old_runs.json.

V3 uses the same graph as V2A (same LIMA run, same pruning); only the grouping differs:
V2A groups by connected components (DIADA's build_mn_datasets), V3 by split groups.

  groups.csv          per benchmark / seed: every V3 group, and whether it holds the target
  target_sets.csv     per benchmark / seed: the MN target dataset under V1, V2A (components)
                      and V3 (union of the target's groups), and the columns V3 leaves out
  utility.csv         a quick model on each target dataset (5-fold cross-validation,
                      HistGradientBoosting; ROC AUC and macro F1 for bank, R^2 for price), plus
                      all columns: do the columns V3 leaves out help predict the target?
                      This is a fast proxy, not ARDA's AutoGluon models.
  png/<name>_split_seed<s>.png   the graph with the target dataset of each grouping marked

Run with:  python explorations/split_grouping_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate  # noqa: E402

from diada import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from two_sided_plots import SOURCES, circles_layout, label  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided"
V1, V2A, V3 = "V1 two-sided", "V2A two-sided + stratified mediation", "V3 V2A + split grouping"


def utility(df: pd.DataFrame, cols, target: str) -> dict:
    X = df[[c for c in cols if c != target]].copy()
    if X.shape[1] == 0:
        return {}
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = df[target]
    if y.nunique() <= 20:
        model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        s = cross_validate(model, X, y, cv=cv, scoring={"auc": "roc_auc", "f1": "f1_macro"})
        return dict(auc=round(s["test_auc"].mean(), 4), f1_macro=round(s["test_f1"].mean(), 4))
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
    s = cross_validate(model, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring={"r2": "r2"})
    return dict(r2=round(s["test_r2"].mean(), 4))


def plot(name, target, seed, v2a, v3) -> None:
    g = nx.Graph()
    g.add_edges_from(k.split("-", 1) for k in v2a["edges"])
    comp, grp = set(v2a["mn_target_cluster"]), set(v3["mn_target_cluster"])
    pos = circles_layout(g)
    xs, ys = zip(*pos.values())
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for k in v2a["edges"]:
        a, c = k.split("-", 1)
        ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], color="#888780", alpha=0.6, linewidth=1.5, zorder=1)
    colors = ["#E24B4A" if n == target else "#EF9F27" if n in grp else "#E6F1FB" if n in comp else "#F1EFE8" for n in g]
    edges = ["#185FA5" if n in comp and n not in grp else "#5F5E5A" for n in g]
    nx.draw_networkx_nodes(g, pos, node_color=colors, edgecolors=edges, linewidths=[2.2 if e == "#185FA5" else 0.8 for e in edges],
                           node_size=1000, ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: label(name, n) for n in g}, font_size=8.5, ax=ax)
    ax.set_title(f"{name}, seed {seed}: graph after pruning (V2A = V3, {len(v2a['edges'])} edges)\n"
                 f"red = target; orange = in V3's target dataset ({len(grp)} columns); "
                 f"blue ring = in the components' target dataset ({len(comp)}) but left out by V3", fontsize=11)
    ax.axis("off")
    lines = [f"V3 groups ({len(v3['groups'])}):"]
    for gr in v3["groups"]:
        mark = "*" if target in gr else " "
        lines.append(f"{mark} {{{', '.join(label(name, c) for c in gr)}}}")
    lines.append("")
    lines.append("* = contains the target")
    tx.text(0, 1, "\n".join(lines[:60]), va="top", family="monospace", fontsize=8.5)
    tx.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "png" / f"{name}_split_seed{seed}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    group_rows, target_rows, util_rows = [], [], []
    for name, (rel, target) in SOURCES.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        util_rows.append(dict(benchmark=name, seed="-", dataset="all columns", columns=df.shape[1] - 1,
                              **utility(df, df.columns, target)))
        for seed in (0, 1):
            v1, v2a, v3 = (runs[(name, v, seed)] for v in (V1, V2A, V3))
            for gr in v3["groups"]:
                group_rows.append(dict(benchmark=name, seed=seed, size=len(gr), has_target=target in gr,
                                       group=" ".join(label(name, c) for c in gr)))
            left_out = sorted(set(v2a["mn_target_cluster"]) - set(v3["mn_target_cluster"]))
            target_rows.append(dict(benchmark=name, seed=seed, groups=len(v3["groups"]),
                                    groups_with_target=sum(target in g for g in v3["groups"]),
                                    V1_components=len(v1["mn_target_cluster"]), V2A_components=len(v2a["mn_target_cluster"]),
                                    V3_split=len(v3["mn_target_cluster"]),
                                    V3_target_dataset=" ".join(label(name, c) for c in v3["mn_target_cluster"]),
                                    left_out_by_V3=" ".join(label(name, c) for c in left_out)))
            for dataset, cols in (("V1 components", v1["mn_target_cluster"]), ("V2A components", v2a["mn_target_cluster"]),
                                  ("V3 split", v3["mn_target_cluster"])):
                util_rows.append(dict(benchmark=name, seed=seed, dataset=dataset, columns=len(cols) - 1, **utility(df, cols, target)))
            plot(name, target, seed, v2a, v3)
            print(f"{name} seed {seed} done", flush=True)
    pd.DataFrame(group_rows).to_csv(OUT / "groups.csv", index=False)
    t = pd.DataFrame(target_rows)
    t.to_csv(OUT / "target_sets.csv", index=False)
    u = pd.DataFrame(util_rows)
    u.to_csv(OUT / "utility.csv", index=False)
    print(t.to_string(index=False))
    print(u.to_string(index=False))


if __name__ == "__main__":
    main()
