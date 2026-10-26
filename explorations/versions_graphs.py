"""
Side-by-side graphs of V0 (jar), V1 (two-sided), V2 (V1 + triangle split, stored as
"V3 V2A + split grouping") and V2.exact (the same split with exact counts over all row pairs,
explorations/exact_counts.py) for every old benchmark, from results/jar_vs_two_sided/old_runs.json.
Every column sits in the same place in all panels (columns with no edge in any version are
left out).

  grey line        edge
  red dashed line  triangle edge A-C split because B explains it (V2, V2.exact)
  red node         target; orange node: in the target's MN dataset

Run with:  python explorations/versions_graphs.py [--seed 0]
Writes results/jar_vs_two_sided/png/<name>_versions4_seed<s>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from diada import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from two_sided_plots import SOURCES, label  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided"
GREY, RED = "#888780", "#E24B4A"
VERSIONS = [("V0 jar", "V0 jar"), ("V1 two-sided", "V1 two-sided"), ("V3 V2A + split grouping", "V2 split"), ("V2.exact", "V2.exact")]
# bank: related columns spread around the circle (same order as bank_three_versions_plot.py)
ORDER = {"bank": ["Class", "V16", "V1", "V11", "V2", "V14", "V4", "V9", "V3", "V15", "V6", "V7", "V13", "V5", "V10", "V8", "V12"]}


def spread_order(nodes, edges) -> list:
    """Place connected columns apart: walk the nodes by degree and alternate them around the circle."""
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    by_degree = sorted(nodes, key=lambda n: (-g.degree(n), str(n)))
    half = (len(by_degree) + 1) // 2
    out = []
    for a, b in zip(by_degree[:half], by_degree[half:] + [None]):
        out.append(a)
        if b is not None:
            out.append(b)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    for name, (_, target) in SOURCES.items():
        panels = [(runs[(name, v, 0 if v == "V0 jar" else args.seed)], title) for v, title in VERSIONS]
        edges = {id(r): {tuple(k.split("-", 1)) for k in r["edges"]} | {(t["a"], t["c"]) for t in r.get("pruned", [])} for r, _ in panels}
        nodes = sorted({c for es in edges.values() for e in es for c in e} | {target})
        order = [c for c in ORDER.get(name, []) if c in nodes] or spread_order(nodes, set().union(*edges.values()))
        order = [target] + [c for c in order if c != target]
        angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / len(order)
        pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(order, angles)}
        small = len(order) > 25

        fig, axes = plt.subplots(1, 4, figsize=(28, 8.6))
        for ax, (r, title) in zip(axes, panels):
            split = {frozenset((t["a"], t["c"])) for t in r.get("pruned", [])}
            for a, c in edges[id(r)]:
                gone = frozenset((a, c)) in split
                ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=RED if gone else GREY,
                        linewidth=1.6 if gone else 1.1, alpha=0.85 if gone else 0.5, linestyle=(0, (4, 3)) if gone else "-")
            mn = set(r["mn_target_cluster"])
            for col, (x, y) in pos.items():
                fill = RED if col == target else "#FAC775" if col in mn else "white"
                ax.scatter([x], [y], s=160 if small else 380, color=fill, edgecolors="#5F5E5A", linewidths=0.7, zorder=2)
                ax.text(x * 1.17, y * 1.17, label(name, col), ha="center", va="center", fontsize=6.5 if small else 8.5,
                        rotation=0)
            n_edges = len(edges[id(r)])
            extra = f", {len(split)} split (red dashed)" if split else ""
            seed = "" if r["version"] == "V0 jar" else f", seed {r['seed']}"
            ax.set_title(f"{title}{seed}: {n_edges} edges{extra}\nMN target dataset: {len(mn)} columns; {r['seconds']:.1f} s",
                         fontsize=11.5)
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.legend(handles=[Line2D([], [], color=GREY, lw=1.5, label="edge"),
                            Line2D([], [], color=RED, lw=1.8, ls=(0, (4, 3)), label="split: A-C explained by B"),
                            Line2D([], [], marker="o", ls="", markerfacecolor=RED, markeredgecolor="#5F5E5A", ms=10, label=f"target ({target})"),
                            Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=10,
                                   label="in the target's MN dataset")],
                   loc="upper center", ncol=4, fontsize=10.5, frameon=False)
        fig.suptitle(name, x=0.02, y=0.99, ha="left", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / "png" / f"{name}_versions4_seed{args.seed}.png", dpi=75)
        plt.close(fig)
        print(f"{name}: written", flush=True)


if __name__ == "__main__":
    main()
