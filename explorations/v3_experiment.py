"""
V0 (DIADA-0.8.jar) vs V1 (Python LIMA, two-sided test) vs V2 (V1 + splitting mediated triangles,
explorations/v2_mediation_split.py) vs V3 (V2 + deciding open paths, explorations/v3_open_paths.py).

Synthetic: the nine three-column benchmarks of explorations/synthetic_triads.py, V1-V3 with seeds
0-4 (V0 is deterministic: once). Scored on the groups each version produces: DIADA's connected
components for V0 and V1, the split groups for V2 and V3.

Old benchmarks: diamonds, bank, house_sales (data/base_datasets) and bank with injected noise
(data/noise_mn/bank_mn_noise.csv), V1-V3 with seeds 0 and 1. Per run: edges, V2 splits, V3
separations (with the tests behind them), groups, the target's MN dataset, time; and a figure
with the four graphs side by side.

Run with:  python explorations/v3_experiment.py [--part A B]
Writes results/v0_v3/{synthetic.csv, old.json, png/<name>_seed<s>.png}
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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
from diada.core.diada_tool import invoke_diada  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lima_two_sided import run  # noqa: E402
from synthetic_triads import EXPECTED, EXPECTED_EDGES  # noqa: E402
from two_sided_plots import label  # noqa: E402
from v2_mediation_split import target_dataset  # noqa: E402
from v3_open_paths import run_v2_v3  # noqa: E402

OUT = paths.RESULTS_DIR / "v0_v3"
EXPECTED_GROUPS = {"independent": "A | B | C", "pair_isolated": "A-B | C", "chain": "A-B | B-C", "common_cause": "A-B | B-C",
                   "hub_join": "A-B | B-C", "direct_triangle": "A-B-C", "sum_collider": "A-B-C", "xor": "n/a",
                   "interaction": "A-B-C"}
OLD = {"diamonds": ("base_datasets/diamonds.csv", "price"), "bank": ("base_datasets/bank.csv", "Class"),
       "house_sales": ("base_datasets/house_sales.csv", "price"), "bank_mn": ("noise_mn/bank_mn_noise.csv", "Class")}
BANK_ORDER = ["Class", "V16", "V1", "V11", "V2", "V14", "V4", "V9", "V3", "V15", "V6", "V7", "V13", "V5", "V10", "V8", "V12"]


def components(graph: nx.Graph):
    return sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda g: (-len(g), g))


def run_all(df: pd.DataFrame, seeds) -> list:
    """V0 once, then V1, V2 and V3 per seed. Each: dict(version, seed, graph (LIMA's graph, before
    any split), groups, splits, separated, tests, open_tests, seconds)."""
    out = []
    start = time.perf_counter()
    g0 = build_soundness_graph(invoke_diada(df, 10, engine="jar", verbose=False), threshold=4)
    out.append(dict(version="V0", seed=0, graph=g0, groups=components(g0), splits=[], separated=[], tests=[],
                    open_tests=[], seconds=time.perf_counter() - start))
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            r = run(df, "two_sided", seed, Path(d))
        g1 = build_soundness_graph(r["soundness"], threshold=4)
        out.append(dict(version="V1", seed=seed, graph=g1, groups=components(g1), splits=[], separated=[], tests=[],
                        open_tests=[], seconds=r["seconds"]))
        v = run_v2_v3(r["lima"], g1)
        out.append(dict(version="V2", seed=seed, graph=g1, groups=v["v2"]["groups"], splits=v["v2"]["splits"], separated=[],
                        tests=v["v2"]["tests"], open_tests=[], seconds=r["seconds"] + v["v2"]["seconds"],
                        seconds_extra=v["v2"]["seconds"]))
        out.append(dict(version="V3", seed=seed, graph=g1, groups=v["v3"]["groups"], splits=v["v3"]["splits"],
                        separated=v["v3"]["separated"], tests=v["v3"]["tests"], open_tests=v["v3"]["open_tests"],
                        seconds=r["seconds"] + v["v3"]["seconds"], seconds_extra=v["v3"]["seconds"]))
    return out


def grouping_text(groups, columns) -> str:
    groups = list(groups) + [[c] for c in columns if not any(c in g for g in groups)]
    return " | ".join(sorted("-".join(sorted(g)) for g in groups))


def part_a() -> None:
    rows = []
    for name in EXPECTED:
        df = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{name}.csv")
        must, may = EXPECTED_EDGES[name]
        for r in run_all(df, range(5)):
            edges = {"-".join(sorted(e)) for e in r["graph"].edges}
            got, want = grouping_text(r["groups"], "ABC"), EXPECTED_GROUPS[name]
            rows.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=" ".join(sorted(edges)) or "none",
                             graph_ok=must <= edges <= must | may,
                             splits=" ".join(f"{t['a']}-{t['c']} at {t['b']}" for t in r["splits"]),
                             separated=" ".join(f"{t['a']}-{t['c']}" for t in r["separated"]),
                             tests=" ".join(f"{t['a']}-{t['c']}|{t['b']}:{t['share_upper']:.2f}" for t in r["tests"]),
                             open_tests=" ".join(f"{t['a']}-{t['c']}|{t['b']}:{t['share_upper']:.2f}"
                                                 f"{'(interaction)' if t['interaction'] else ''}" for t in r["open_tests"]),
                             grouping=got, expected=want, ok=None if want == "n/a" else got == want,
                             seconds=round(r["seconds"], 3), seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        print(f"A: {name} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "synthetic.csv", index=False)


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
    grey, red, purple = "#888780", "#E24B4A", "#7F77DD"
    nodes = sorted({c for r in panels for e in r["graph"].edges for c in e} | {target})
    order = [c for c in BANK_ORDER if c in nodes] if name == "bank" else spread_order(nodes, set().union(*(set(r["graph"].edges) for r in panels)))
    order = [target] + [c for c in order if c != target]
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / len(order)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(order, angles)}
    small = len(order) > 25
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 8.2))
    for ax, r in zip(axes, panels):
        split = {frozenset((t["a"], t["c"])) for t in r["splits"]}
        for a, c in r["graph"].edges:
            gone = frozenset((a, c)) in split
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=red if gone else grey,
                    linewidth=1.7 if gone else 1.1, alpha=0.85 if gone else 0.5, linestyle=(0, (4, 3)) if gone else "-")
        for t in r["separated"]:
            a, c = t["a"], t["c"]
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=purple, linewidth=1.6, alpha=0.9,
                    linestyle=(0, (1, 2)))
        mn = set(target_dataset(r["groups"], target))
        for col, (x, y) in pos.items():
            fill = red if col == target else "#FAC775" if col in mn else "white"
            ax.scatter([x], [y], s=150 if small else 360, color=fill, edgecolors="#5F5E5A", linewidths=0.7, zorder=2)
            ax.text(x * 1.17, y * 1.17, label(name, col), ha="center", va="center", fontsize=6.3 if small else 8.3)
        seed = "" if r["version"] == "V0" else f", seed {r['seed']}"
        extra = (f", {len(split)} split" if split else "") + (f", {len(r['separated'])} open paths cut" if r["separated"] else "")
        ax.set_title(f"{r['version']}{seed}: {r['graph'].number_of_edges()} edges{extra}\n"
                     f"MN target dataset: {len(mn)} columns; {r['seconds']:.1f} s", fontsize=11)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=grey, lw=1.5, label="edge"),
                        Line2D([], [], color=red, lw=1.8, ls=(0, (4, 3)), label="V2: triangle edge A-C split (explained by B)"),
                        Line2D([], [], color=purple, lw=1.8, ls=(0, (1, 2)), label="V3: open path A-B-C cut (A, C unrelated given B)"),
                        Line2D([], [], marker="o", ls="", markerfacecolor=red, markeredgecolor="#5F5E5A", ms=10, label=f"target ({label(name, target)})"),
                        Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=10,
                               label="in the target's MN dataset")],
               loc="upper center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle(name, x=0.02, y=0.99, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=72)
    plt.close(fig)


def part_b() -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    out = []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        runs = run_all(df, (0, 1))
        for r in runs:
            g = r["graph"]
            out.append(dict(benchmark=name, version=r["version"], seed=r["seed"], cols=df.shape[1], rows=len(df),
                            edges=sorted("-".join(sorted(e)) for e in g.edges), un_kept=sorted(g.nodes),
                            groups=r["groups"], mn_target=target_dataset(r["groups"], target),
                            splits=r["splits"], separated=r["separated"], tests=r["tests"], open_tests=r["open_tests"],
                            seconds=round(r["seconds"], 3), seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        v0 = next(r for r in runs if r["version"] == "V0")
        for seed in (0, 1):
            panels = [v0] + [r for r in runs if r["version"] != "V0" and r["seed"] == seed]
            plot(name, target, panels, OUT / "png" / f"{name}_seed{seed}.png")
        print(f"B: {name} done", flush=True)
    (OUT / "old.json").write_text(json.dumps(out, indent=1))


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
