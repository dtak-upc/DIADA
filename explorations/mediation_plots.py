"""
Plots and a check of the edges removed by V2 / V2.1 / V2A (explorations/lima_mediation.py) on the
old benchmarks, from results/jar_vs_two_sided/old_runs.json (two_sided_experiment.py --v0 jar).

For every dataset and seed:
  <name>_versions_seed<s>.png       DIADA's own cluster maps for V0 jar, V1, V2, V2.1 and V2A
  <name>_pruned_seed<s>.png         V1's graph (V2 runs the same LIMA with the same seed) with the
                                    edges V2 removed in red dashed; target in red
  <name>_pruned_v2a_seed<s>.png     the same for V2A
and results/jar_vs_two_sided/pruned_edges.csv: every edge V2, V2.1 or V2A removed, with the
column that explains it, the conditional evidence LIMA's statistic gave, and a check on
the full data: the information A and C share (bits, bucketed as DIADA does) overall and
once the explaining column B is known. If B really explains A-C, the second is ~0.

Run with:  python explorations/mediation_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from two_sided_plots import SOURCES, circles_layout, entropies, label  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided"
V2, V21, V2A = "V2 two-sided + mediation", "V2.1 one-sided + mediation", "V2A two-sided + stratified mediation"


def main() -> None:
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    rows = []
    for name, (rel, target) in SOURCES.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        codes, h = entropies(df)
        for seed in (0, 1):
            panels = [runs[(name, "V0 jar", 0)], runs[(name, "V1 two-sided", seed)], runs[(name, V2, seed)],
                      runs[(name, V21, seed)], runs[(name, V2A, seed)]]

            fig, axes = plt.subplots(2, 3, figsize=(30, 17))
            axes.flat[-1].axis("off")
            for ax, r in zip(axes.flat, panels):
                ax.imshow(mpimg.imread(paths.REPO_ROOT / r["png"]))
                ax.axis("off")
                ax.set_title(f"{name} - {r['version']}: {len(r['edges'])} edges" + ("" if r["version"] == "V0 jar" else f", seed {seed}"), fontsize=15)
            fig.tight_layout()
            fig.savefig(OUT / "png" / f"{name}_versions_seed{seed}.png", dpi=70)
            plt.close(fig)

            for version in (V2, V21, V2A):
                for t in runs[(name, version, seed)]["pruned"]:
                    a, b, c = t["a"], t["mediator"], t["c"]
                    mi = h(codes[a]) + h(codes[c]) - h(codes[a], codes[c])
                    cmi = h(codes[a], codes[b]) + h(codes[c], codes[b]) - h(codes[a], codes[b], codes[c]) - h(codes[b])
                    rows.append(dict(benchmark=name, seed=seed, version=version.split(" ")[0],
                                     edge=f"{label(name, a)}-{label(name, c)}", explained_by=label(name, b),
                                     conditional_z=t["evidence"], mi_bits=round(mi, 4), mi_given_mediator=round(cmi, 4),
                                     share_left=round(cmi / mi, 2) if mi > 1e-4 else None))

            for version, suffix in ((V2, ""), (V2A, "_v2a")):
                pruned_plot(name, target, seed, runs[(name, "V1 two-sided", seed)], runs[(name, version, seed)], suffix)
            print(f"{name} seed {seed}: plots written", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pruned_edges.csv", index=False)


def pruned_plot(name, target, seed, v1, v2, suffix) -> None:
    """V1's graph with the edges `v2` (V2 or V2A, same LIMA run) removed in red dashed."""
    pruned = {frozenset((t["a"], t["c"])): t["mediator"] for t in v2["pruned"]}
    g = nx.Graph()
    g.add_edges_from(k.split("-", 1) for k in v1["edges"])
    pos = circles_layout(g)
    xs, ys = zip(*pos.values())
    fig, ax = plt.subplots(figsize=(min(18, 2.2 * (max(xs) - min(xs) + 3)), min(12, 2.2 * (max(ys) - min(ys) + 3) + 1)))
    for k in v1["edges"]:
        a, c = k.split("-", 1)
        gone = frozenset((a, c)) in pruned
        ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, alpha=0.85 if gone else 0.6,
                color="#E24B4A" if gone else "#888780", linewidth=2.2 if gone else 1.6,
                linestyle=(0, (4, 3)) if gone else "-")
    nx.draw_networkx_nodes(g, pos, node_color=["#E24B4A" if n == target else "#F1EFE8" for n in g],
                           edgecolors="#5F5E5A", linewidths=0.8, node_size=900, ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: label(name, n) for n in g}, font_size=9, ax=ax)
    ax.set_title(f"{name}, seed {seed}: V1 graph ({len(v1['edges'])} edges); red dashed = removed by {v2['version'].split(' ')[0]} "
                 f"as fully explained by a third column ({len(pruned)})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "png" / f"{name}_pruned{suffix}_seed{seed}.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
