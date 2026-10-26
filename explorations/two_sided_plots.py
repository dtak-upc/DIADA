"""
Plots and edge tables for the V0 vs. V1 (two-sided) comparison on the old benchmarks,
from <out>/old_runs.json (written by two_sided_experiment.py with the same --v0):
    --v0 jar   V0 = DIADA-0.8.jar, out = results/jar_vs_two_sided (the jar runs once, so
               each V1 run is compared with that single jar run)
    --v0 base  V0 = the Python LIMA port, out = results/two_sided

For every dataset and seed:
  <name>_compare_seed<s>.png  DIADA's own cluster maps (dataset_builder.visualize_clusters)
                              for V0 and V1 side by side
  <name>_diff_seed<s>.png     both graphs on one layout: edges found by both versions in
                              grey, only by V0 in orange, only by V1 in blue (orange/blue
                              dashed when their normalised shared information is below
                              0.05); line width grows with normalised shared information;
                              target in red
and <out>/old_edges.csv: every edge of either version with its mutual
information (bits) and its normalised mutual information = MI / min(H(x), H(y)), the
share of the maximum the two columns could possibly share.

Run with:  python explorations/two_sided_plots.py [--v0 jar|base]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402
from diada.core.diada_tool import _label_columns  # noqa: E402

OUT_DIRS = {"base": paths.RESULTS_DIR / "two_sided", "jar": paths.RESULTS_DIR / "jar_vs_two_sided"}
V0_LABEL = {"base": "V0 base", "jar": "V0 jar"}
SOURCES = {"diamonds": ("base_datasets/diamonds.csv", "price"), "bank": ("base_datasets/bank.csv", "Class"),
           "house_sales": ("base_datasets/house_sales.csv", "price"), "bank_mn": ("noise_mn/bank_mn_noise.csv", "Class")}
# bank's anonymised columns, identified from their values (the UCI bank-marketing dataset)
BANK = {"V1": "age", "V2": "job", "V3": "marital", "V4": "education", "V5": "default", "V6": "balance",
        "V7": "housing_loan", "V8": "personal_loan", "V9": "contact", "V10": "day", "V11": "month", "V12": "duration",
        "V13": "campaign", "V14": "pdays", "V15": "previous", "V16": "poutcome"}


def label(name: str, col: str) -> str:
    return BANK.get(col, col) if name.startswith("bank") else col


def entropies(df: pd.DataFrame):
    lab = _label_columns(df.copy(), 10)
    codes = {c: pd.factorize(lab.iloc[:, i].astype(str))[0] for i, c in enumerate(df.columns)}

    def h(*cs):
        j = cs[0].astype(np.int64)
        for c in cs[1:]:
            j = j * (int(c.max()) + 1) + c
        cnt = np.unique(j, return_counts=True)[1]
        p = cnt / cnt.sum()
        return (-(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * cnt.sum())) / np.log(2)

    return codes, h


def circles_layout(g: nx.Graph, max_row_width: float = 14.0) -> dict:
    """Each connected group on its own circle (radius growing with its size), groups placed
    left to right in rows, largest first, so every edge is a visible chord of its circle."""
    pos, x, y, row_height = {}, 0.0, 0.0, 0.0
    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        nodes = sorted(comp)
        r = 0.5 if len(nodes) <= 2 else 0.9 + 0.22 * len(nodes)
        if x > 0 and x + 2 * r > max_row_width:
            x, y, row_height = 0.0, y - row_height - 1.2, 0.0
        for i, n in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes) + np.pi / 2
            pos[n] = (x + r + r * np.cos(angle), y - r + r * np.sin(angle))
        x += 2 * r + 1.2
        row_height = max(row_height, 2 * r)
    return pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0", choices=["jar", "base"], default="jar")
    args = parser.parse_args()
    OUT = OUT_DIRS[args.v0]
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    rows = []
    for name, (rel, target) in SOURCES.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        codes, h = entropies(df)
        for seed in (0, 1):
            v0 = runs[(name, V0_LABEL[args.v0], 0 if args.v0 == "jar" else seed)]
            v1 = runs[(name, "V1 two-sided", seed)]

            # DIADA's own maps, side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
            for ax, r in zip(axes, (v0, v1)):
                ax.imshow(mpimg.imread(paths.REPO_ROOT / r["png"]))
                ax.axis("off")
                ax.set_title(f"{name} - {r['version']}: {len(r['edges'])} edges" + ("" if r["version"] == "V0 jar" else f", seed {seed}"), fontsize=15)
            fig.tight_layout()
            fig.savefig(OUT / "png" / f"{name}_compare_seed{seed}.png", dpi=80)
            plt.close(fig)

            # edge table with normalised shared information
            nmi = {}
            for k in set(v0["edges"]) | set(v1["edges"]):
                a, b = k.split("-", 1)
                mi = h(codes[a]) + h(codes[b]) - h(codes[a], codes[b])
                nmi[k] = max(0.0, mi) / max(1e-9, min(h(codes[a]), h(codes[b])))
                rows.append(dict(benchmark=name, seed=seed, edge="-".join(label(name, p) for p in (a, b)),
                                 in_V0=k in v0["edges"], in_V1=k in v1["edges"], mi_bits=round(mi, 4),
                                 normalised_mi=round(nmi[k], 3), V0_value=v0["edges"].get(k, {}).get("value"),
                                 V1_value=v1["edges"].get(k, {}).get("value")))

            # difference plot on one layout
            g = nx.Graph()
            for k in set(v0["edges"]) | set(v1["edges"]):
                g.add_edge(*k.split("-", 1))
            pos = circles_layout(g)
            xs, ys = zip(*pos.values())
            width, height = max(xs) - min(xs) + 3, max(ys) - min(ys) + 3
            fig, ax = plt.subplots(figsize=(min(18, 2.2 * width), min(12, 2.2 * height + 1)))
            for k in set(v0["edges"]) | set(v1["edges"]):
                a, b = k.split("-", 1)
                in0, in1 = k in v0["edges"], k in v1["edges"]
                both = in0 and in1
                color = "#888780" if both else ("#D85A30" if in0 else "#378ADD")
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], zorder=1, alpha=0.8,
                        color=color, linewidth=1.4 + 6 * min(nmi[k], 1.0),
                        linestyle="-" if both or nmi[k] >= 0.05 else (0, (4, 3)))
            colors = ["#E24B4A" if n == target else "#F1EFE8" for n in g]
            nx.draw_networkx_nodes(g, pos, node_color=colors, edgecolors="#5F5E5A", linewidths=0.8, node_size=900, ax=ax)
            nx.draw_networkx_labels(g, pos, labels={n: label(name, n) for n in g}, font_size=9, ax=ax)
            e0, e1 = set(v0["edges"]), set(v1["edges"])
            ax.set_title(f"{name}, {v0['version']} vs V1 seed {seed}: grey = both ({len(e0 & e1)}), "
                         f"orange = only {v0['version']} ({len(e0 - e1)}), blue = only V1 ({len(e1 - e0)}); "
                         f"dashed = normalised MI < 0.05, width = normalised MI", fontsize=11)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(OUT / "png" / f"{name}_diff_seed{seed}.png", dpi=90)
            plt.close(fig)
            print(f"{name} seed {seed}: plots written", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "old_edges.csv", index=False)


if __name__ == "__main__":
    main()
