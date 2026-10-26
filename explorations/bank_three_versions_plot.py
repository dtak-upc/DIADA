"""
Side-by-side graphs for bank: V0 (jar), V1 (Python LIMA, two-sided test), V2 (V1 + splitting a
triangle A-B-C into {A, B} and {B, C} when B explains A-C; stored as "V3 V2A + split grouping").
All panels use the same position for every column.

  V0 panel: grey = edge V1 also has, orange = only V0
  V1 panel: grey = edge V0 also has, blue = only V1
  V2 panel: V1's graph; red dashed = the A-C edge of a triangle that V2 split at B
  Nodes: red = target (Class); orange = in the target's MN dataset; white = elsewhere

Run with:  python explorations/bank_three_versions_plot.py
Writes results/jar_vs_two_sided/png/bank_three_versions_seed<s>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from diada import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from two_sided_plots import BANK  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided"
GREY, ORANGE, BLUE, RED = "#888780", "#D85A30", "#378ADD", "#E24B4A"
# related columns spread around the circle so their edges do not overlap
COLUMNS = ["Class", "V16", "V1", "V11", "V2", "V14", "V4", "V9", "V3", "V15", "V6", "V7", "V13", "V5", "V10", "V8", "V12"]


def positions() -> dict:
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(COLUMNS)) / len(COLUMNS)
    return {c: (np.cos(a), np.sin(a)) for c, a in zip(COLUMNS, angles)}


def draw(ax, pos, edges: dict, target_set, title: str) -> None:
    for (a, c), style in edges.items():
        color, dashed = style
        ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], color=color, zorder=1,
                linewidth=2.0 if color != GREY else 1.3, alpha=0.9 if color != GREY else 0.55,
                linestyle=(0, (4, 3)) if dashed else "-")
    for col, (x, y) in pos.items():
        fill = RED if col == "Class" else "#FAC775" if col in target_set else "white"
        ax.scatter([x], [y], s=520, color=fill, edgecolors="#5F5E5A", linewidths=0.8, zorder=2)
        ax.text(x * 1.2, y * 1.2, BANK.get(col, col), ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    pos = positions()
    for seed in (0, 1):
        v0, v1, v2 = runs[("bank", "V0 jar", 0)], runs[("bank", "V1 two-sided", seed)], runs[("bank", "V3 V2A + split grouping", seed)]
        e0 = {tuple(k.split("-", 1)) for k in v0["edges"]}
        e1 = {tuple(k.split("-", 1)) for k in v1["edges"]}
        s0, s1 = {frozenset(e) for e in e0}, {frozenset(e) for e in e1}
        split = {frozenset((t["a"], t["c"])): t["mediator"] for t in v2["pruned"]}

        fig, axes = plt.subplots(1, 3, figsize=(21, 8))
        draw(axes[0], pos, {e: (GREY if frozenset(e) in s1 else ORANGE, False) for e in e0}, set(v0["mn_target_cluster"]),
             f"V0 jar: {len(e0)} edges ({len(s0 - s1)} only in V0)\nMN target dataset: {len(v0['mn_target_cluster'])} columns")
        draw(axes[1], pos, {e: (GREY if frozenset(e) in s0 else BLUE, False) for e in e1}, set(v1["mn_target_cluster"]),
             f"V1 two-sided, seed {seed}: {len(e1)} edges ({len(s1 - s0)} only in V1)\nMN target dataset: {len(v1['mn_target_cluster'])} columns")
        draw(axes[2], pos, {e: ((RED, True) if frozenset(e) in split else (GREY, False)) for e in e1}, set(v2["mn_target_cluster"]),
             f"V2 split, seed {seed}: {len(split)} of V1's {len(e1)} edges split (red dashed)\n"
             f"MN target dataset: {len(v2['mn_target_cluster'])} columns")
        names = ["{" + ", ".join(BANK.get(c, c) for c in g) + "}" for g in v2["groups"]]
        groups = "   ".join(names)
        lines = ["   ".join(names[i:i + 4]) for i in range(0, len(names), 4)]
        fig.text(0.5, 0.01, "V2 groups:  " + "\n".join(lines), ha="center", va="bottom", fontsize=9.5)
        fig.legend(handles=[Line2D([], [], color=GREY, lw=1.5, label="edge in V0 and V1"),
                            Line2D([], [], color=ORANGE, lw=2, label="only V0"),
                            Line2D([], [], color=BLUE, lw=2, label="only V1"),
                            Line2D([], [], color=RED, lw=2, ls=(0, (4, 3)), label="V2: split (A-C explained by B)"),
                            Line2D([], [], marker="o", ls="", markerfacecolor=RED, markeredgecolor="#5F5E5A", ms=11, label="target (Class)"),
                            Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=11,
                                   label="in the target's MN dataset")],
                   loc="upper center", ncol=6, fontsize=10, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 0.93))
        fig.savefig(OUT / "png" / f"bank_three_versions_seed{seed}.png", dpi=80)
        plt.close(fig)
        print(f"seed {seed}: V2 groups = {groups}")


if __name__ == "__main__":
    main()
