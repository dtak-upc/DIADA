"""
V1 rebuilt on the new V0 (diada.core.LIMA, the Python port of the jar) vs the original V1 (built on
the old Python LIMA, results/v0_v3).

  V0          diada.core.LIMA = the jar's output (seed 42 is the jar's seed; also seeds 0-4 to see
              how much the seed alone moves things)
  V1 new      explorations/v1_two_sided.py: V0 with the two-sided test
  V1 old      the stored results of the previous V1 (explorations/lima_two_sided.py on the old
              Python LIMA), seeds 0-4 (synthetic) and 0-1 (old benchmarks), from results/v0_v3

Synthetic: the nine three-column benchmarks (explorations/synthetic_triads.py), scored on DIADA's
graph (edges within must/may) and on its groups (connected components) against the expected ones.
Old benchmarks: diamonds, bank, house_sales, bank_mn; per run the edges, how they compare with the
jar's, the target's MN dataset (its connected component), time. Seeds 42, 0, 1.

Run with:  python explorations/v1_experiment.py [--part A B]
Writes results/v1_new/{synthetic.csv, synthetic_summary.csv, old.json, old.csv, edges.csv,
png/<name>.png}
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
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402
from diada.core.dataset_builder import build_soundness_graph  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_triads import EXPECTED, EXPECTED_EDGES  # noqa: E402
from two_sided_plots import label  # noqa: E402
from v1_two_sided import run  # noqa: E402

OUT = paths.RESULTS_DIR / "v1_new"
STORED = paths.RESULTS_DIR / "v0_v3"
EXPECTED_GROUPS = {"independent": "A | B | C", "pair_isolated": "A-B | C", "chain": "A-B | B-C", "common_cause": "A-B | B-C",
                   "hub_join": "A-B | B-C", "direct_triangle": "A-B-C", "sum_collider": "A-B-C", "xor": "n/a",
                   "interaction": "A-B-C"}
OLD = {"diamonds": ("base_datasets/diamonds.csv", "price"), "bank": ("base_datasets/bank.csv", "Class"),
       "house_sales": ("base_datasets/house_sales.csv", "price"), "bank_mn": ("noise_mn/bank_mn_noise.csv", "Class")}
BANK_ORDER = ["Class", "V16", "V1", "V11", "V2", "V14", "V4", "V9", "V3", "V15", "V6", "V7", "V13", "V5", "V10", "V8", "V12"]
SYNTH_SEEDS, OLD_SEEDS = (42, 0, 1, 2, 3, 4), (42, 0, 1)


def edge_names(graph: nx.Graph) -> list:
    return sorted("-".join(sorted(e)) for e in graph.edges)


def grouping_text(graph: nx.Graph, columns) -> str:
    groups = [sorted(c) for c in nx.connected_components(graph)]
    groups += [[c] for c in columns if c not in graph]
    return " | ".join(sorted("-".join(g) for g in groups))


def component(graph: nx.Graph, target: str) -> list:
    return sorted(nx.node_connected_component(graph, target)) if target in graph else [target]


def part_a() -> None:
    stored = pd.read_csv(STORED / "synthetic.csv")
    rows = []
    for name in EXPECTED:
        df = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{name}.csv")
        must, may = EXPECTED_EDGES[name]
        want = EXPECTED_GROUPS[name]
        for version, two_sided in (("V0", False), ("V1 new", True)):
            for seed in SYNTH_SEEDS:
                r = run(df, two_sided=two_sided, seed=seed)
                g = build_soundness_graph(r["soundness"], threshold=4)
                edges = set(edge_names(g))
                got = grouping_text(g, "ABC")
                rows.append(dict(benchmark=name, version=version, seed=seed, edges=" ".join(sorted(edges)) or "none",
                                 graph_ok=must <= edges <= must | may, grouping=got, expected=want,
                                 ok=None if want == "n/a" else got == want, seconds=round(r["seconds"], 3)))
        for _, s in stored[(stored.benchmark == name) & (stored.version == "V1")].iterrows():
            rows.append(dict(benchmark=name, version="V1 old", seed=int(s.seed), edges=s.edges, graph_ok=bool(s.graph_ok),
                             grouping=s.grouping, expected=want, ok=None if want == "n/a" else s.grouping == want,
                             seconds=s.seconds))
        print(f"A: {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "synthetic.csv", index=False)
    summary = out.groupby(["benchmark", "version"], sort=False).agg(
        runs=("seed", "size"), graph_ok=("graph_ok", "sum"), grouping_ok=("ok", lambda s: s.sum() if s.notna().all() else np.nan),
        edges=("edges", lambda s: " / ".join(f"{e} ({n})" for e, n in s.value_counts().items()))).reset_index()
    summary.to_csv(OUT / "synthetic_summary.csv", index=False)


def spread_order(nodes, edges) -> list:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    by_degree = sorted(nodes, key=lambda n: (-g.degree(n), str(n)))
    half = (len(by_degree) + 1) // 2
    out = []
    for a, b in zip(by_degree[:half], by_degree[half:] + [None]):
        out += [a] if b is None else [a, b]
    return out


def plot(name: str, target: str, panels: list, path: Path) -> None:
    """panels: (title, edges as "A-B" strings, MN target dataset, seconds); V0's edges are drawn grey,
    an edge V0 lacks blue, a V0 edge missing from the panel red dashed"""
    grey, blue, red = "#888780", "#378ADD", "#E24B4A"
    split = lambda e: tuple(e.split("-", 1))
    v0 = set(panels[0][1])
    nodes = sorted({c for _, es, _, _ in panels for e in es for c in split(e)} | {target})
    order = [c for c in BANK_ORDER if c in nodes] if name == "bank" else spread_order(
        nodes, {split(e) for _, es, _, _ in panels for e in es})
    order = [target] + [c for c in order if c != target]
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / len(order)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(order, angles)}
    small = len(order) > 25
    fig, axes = plt.subplots(2, 3, figsize=(20, 14.5))
    for i, (ax, (title, es, mn, seconds)) in enumerate(zip(axes.flat, panels)):
        es = set(es)
        draw = [(e, grey, "-", 0.5) for e in es & v0] + [(e, blue, "-", 0.85) for e in es - v0]
        draw += [(e, red, (0, (4, 3)), 0.8) for e in v0 - es]
        for e, color, style, alpha in draw:
            a, c = split(e)
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=color, linestyle=style,
                    linewidth=1.1 if color == grey else 1.5, alpha=alpha)
        for col, (x, y) in pos.items():
            fill = red if col == target else "#FAC775" if col in mn else "white"
            ax.scatter([x], [y], s=150 if small else 360, color=fill, edgecolors="#5F5E5A", linewidths=0.7, zorder=2)
            ax.text(x * 1.17, y * 1.17, label(name, col), ha="center", va="center", fontsize=6.3 if small else 8.3)
        extra = "" if i == 0 else f" ({len(es & v0)} shared with V0, +{len(es - v0)}, -{len(v0 - es)})"
        ax.set_title(f"{title}: {len(es)} edges{extra}\nMN target dataset: {len(mn)} columns; {seconds:.1f} s", fontsize=11)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=grey, lw=1.5, label="edge also in V0 (the jar)"),
                        Line2D([], [], color=blue, lw=1.8, label="edge V0 does not have"),
                        Line2D([], [], color=red, lw=1.8, ls=(0, (4, 3)), label="V0 edge this version lacks"),
                        Line2D([], [], marker="o", ls="", markerfacecolor=red, markeredgecolor="#5F5E5A", ms=10,
                               label=f"target ({label(name, target)})"),
                        Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=10,
                               label="in the target's MN dataset")],
               loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, fontsize=10.5, frameon=False)
    fig.suptitle(f"{name}: top row V0 and the old V1 (old Python LIMA); bottom row the new V1 (on the port of the jar)",
                 x=0.02, y=0.995, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def part_b() -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    stored = [r for r in json.loads((STORED / "old.json").read_text()) if r["version"] in ("V0", "V1")]
    out, summary, diffs = [], [], []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        runs = []
        for version, two_sided in (("V0", False), ("V1 new", True)):
            for seed in OLD_SEEDS:
                r = run(df, two_sided=two_sided, seed=seed)
                g = build_soundness_graph(r["soundness"], threshold=4)
                runs.append(dict(benchmark=name, version=version, seed=seed, edges=edge_names(g), mn_target=component(g, target),
                                 seconds=round(r["seconds"], 3), patterns=r["patterns"], pairs_sampled=r["pairs_sampled"]))
        for s in stored:
            if s["benchmark"] == name:
                runs.append(dict(benchmark=name, version="V0 jar (stored)" if s["version"] == "V0" else "V1 old", seed=s["seed"],
                                 edges=s["edges"], mn_target=s["mn_target"], seconds=s["seconds"]))
        jar = next(r for r in runs if r["version"] == "V0" and r["seed"] == 42)
        stored_jar = next(r for r in runs if r["version"] == "V0 jar (stored)")
        assert set(jar["edges"]) == set(stored_jar["edges"]), f"{name}: V0 seed 42 differs from the stored jar run"
        e0, mn0 = set(jar["edges"]), set(jar["mn_target"])
        for r in runs:
            e, mn = set(r["edges"]), set(r["mn_target"])
            summary.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=len(e), shared_with_jar=len(e & e0),
                                only_jar=len(e0 - e), only_this=len(e - e0), overlap=round(len(e & e0) / max(1, len(e | e0)), 3),
                                mn_size=len(mn), mn_overlap_jar=round(len(mn & mn0) / len(mn | mn0), 3), seconds=r["seconds"],
                                patterns=r.get("patterns")))
        # which edges each V1 adds to / drops from the jar's, and in how many of its runs
        for version in ("V1 new", "V1 old"):
            vr = [r for r in runs if r["version"] == version]
            for e in sorted(set().union(*(set(r["edges"]) for r in vr)) | e0):
                n = sum(e in r["edges"] for r in vr)
                if (e in e0) != (n == len(vr)) or 0 < n < len(vr):
                    diffs.append(dict(benchmark=name, version=version, edge=e, in_jar=e in e0, runs_with_edge=n, runs=len(vr)))
        out += runs
        get = lambda v, s: next(r for r in runs if r["version"] == v and r["seed"] == s)
        panels = [(f"V0 = the jar", jar["edges"], jar["mn_target"], stored_jar["seconds"])]
        panels += [(f"V1 old, seed {s}", get("V1 old", s)["edges"], get("V1 old", s)["mn_target"], get("V1 old", s)["seconds"]) for s in (0, 1)]
        panels += [(f"V1 new, seed {s}", get("V1 new", s)["edges"], get("V1 new", s)["mn_target"], get("V1 new", s)["seconds"]) for s in OLD_SEEDS]
        plot(name, target, panels, OUT / "png" / f"{name}.png")
        print(f"B: {name} done", flush=True)
    (OUT / "old.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(summary).to_csv(OUT / "old.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "edges.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", nargs="+", default=["A", "B"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "A" in args.part:
        part_a()
    if "B" in args.part:
        part_b()


if __name__ == "__main__":
    main()
