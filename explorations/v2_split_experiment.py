"""
V2 rebuilt on the new V1 (explorations/v2_split.py) vs the original V2 (on the old Python LIMA,
stored in results/v0_v3).

  V0       diada.core.LIMA = the jar's output (seed 42 is the jar's seed)
  V1 new   V0 with the two-sided test (explorations/v1_two_sided.py)
  V2 new   V1 new + splitting the triangles B explains (explorations/v2_split.py), on the same run
  V1 old, V2 old   stored results of explorations/lima_two_sided.py + v2_mediation_split.py on the
                   old Python LIMA (seeds 0-4 synthetic, 0-1 old benchmarks)

Synthetic: the nine three-column benchmarks (explorations/synthetic_triads.py), scored on the groups
(V0, V1: DIADA's connected components; V2: the split groups) against the expected ones. Seeds 42
and 0-4.
Old benchmarks: diamonds, bank, house_sales, bank_mn. Per run: edges, splits (with their tests),
groups, the target's MN dataset, time. Seeds 42, 0, 1.

Run with:  python explorations/v2_split_experiment.py [--part A B]
Writes results/v2_new/{synthetic.csv, synthetic_summary.csv, old.json, old.csv, png/<name>.png}
"""

from __future__ import annotations

import argparse
import json
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_triads import EXPECTED, EXPECTED_EDGES  # noqa: E402
from two_sided_plots import label  # noqa: E402
from v1_experiment import BANK_ORDER, EXPECTED_GROUPS, OLD, spread_order  # noqa: E402
from v1_two_sided import run  # noqa: E402
from v2_split import run_v2, target_dataset  # noqa: E402

OUT = paths.RESULTS_DIR / "v2_new"
STORED = paths.RESULTS_DIR / "v0_v3"
SYNTH_SEEDS, OLD_SEEDS = (42, 0, 1, 2, 3, 4), (42, 0, 1)


def components(graph: nx.Graph) -> list:
    return sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda g: (-len(g), g))


def grouping_text(groups, columns) -> str:
    groups = list(groups) + [[c] for c in columns if not any(c in g for g in groups)]
    return " | ".join(sorted("-".join(sorted(g)) for g in groups))


def run_versions(df: pd.DataFrame, seeds) -> list:
    """V0 and V1/V2 per seed: dict(version, seed, graph (LIMA's graph, before any split), groups,
    splits, tests, seconds, seconds_extra)"""
    out = []
    for seed in seeds:
        r0 = run(df, two_sided=False, seed=seed)
        g0 = build_soundness_graph(r0["soundness"], threshold=4)
        out.append(dict(version="V0", seed=seed, graph=g0, groups=components(g0), splits=[], tests=[], seconds=r0["seconds"]))
        r1 = run(df, two_sided=True, seed=seed)
        g1 = build_soundness_graph(r1["soundness"], threshold=4)
        out.append(dict(version="V1 new", seed=seed, graph=g1, groups=components(g1), splits=[], tests=[], seconds=r1["seconds"]))
        v2 = run_v2(r1["lima"], g1)
        out.append(dict(version="V2 new", seed=seed, graph=g1, groups=v2["groups"], splits=v2["splits"], tests=v2["tests"],
                        seconds=r1["seconds"] + v2["seconds"], seconds_extra=v2["seconds"]))
    return out


def part_a() -> None:
    stored = pd.read_csv(STORED / "synthetic.csv")
    rows = []
    for name in EXPECTED:
        df = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{name}.csv")
        must, may = EXPECTED_EDGES[name]
        want = EXPECTED_GROUPS[name]
        for r in run_versions(df, SYNTH_SEEDS):
            edges = {"-".join(sorted(e)) for e in r["graph"].edges}
            got = grouping_text(r["groups"], "ABC")
            rows.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=" ".join(sorted(edges)) or "none",
                             graph_ok=must <= edges <= must | may,
                             splits=" ".join(f"{t['a']}-{t['c']} at {t['b']}" for t in r["splits"]),
                             tests=" ".join(f"{t['a']}-{t['c']}|{t['b']}:{t['share_upper']:.2f}" for t in r["tests"]),
                             grouping=got, expected=want, ok=None if want == "n/a" else got == want,
                             seconds=round(r["seconds"], 3), seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        for _, s in stored[(stored.benchmark == name) & stored.version.isin(["V1", "V2"])].iterrows():
            rows.append(dict(benchmark=name, version=f"{s.version} old", seed=int(s.seed), edges=s.edges, graph_ok=bool(s.graph_ok),
                             splits=s.splits if isinstance(s.splits, str) else "", tests=s.tests if isinstance(s.tests, str) else "",
                             grouping=s.grouping, expected=want, ok=None if want == "n/a" else s.grouping == want,
                             seconds=s.seconds, seconds_extra=s.seconds_extra))
        print(f"A: {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "synthetic.csv", index=False)
    order = {"V0": 0, "V1 old": 1, "V1 new": 2, "V2 old": 3, "V2 new": 4}
    summary = out.groupby(["benchmark", "version"], sort=False).agg(
        runs=("seed", "size"), grouping_ok=("ok", lambda s: s.sum() if s.notna().all() else np.nan),
        groupings=("grouping", lambda s: " / ".join(f"{g} ({n})" for g, n in s.value_counts().items())),
        extra_seconds=("seconds_extra", "mean")).reset_index()
    summary["o"] = summary.version.map(order)
    summary.sort_values(["benchmark", "o"], key=lambda c: c.map(list(EXPECTED).index) if c.name == "benchmark" else c) \
        .drop(columns="o").to_csv(OUT / "synthetic_summary.csv", index=False)


def plot(name: str, target: str, panels: list, path: Path) -> None:
    """panels: (title, edges as "A-B" strings, split edges, MN target dataset, seconds)"""
    grey, red = "#888780", "#E24B4A"
    split = lambda e: tuple(e.split("-", 1))
    nodes = sorted({c for _, es, _, _, _ in panels for e in es for c in split(e)} | {target})
    order = [c for c in BANK_ORDER if c in nodes] if name == "bank" else spread_order(
        nodes, {split(e) for _, es, _, _, _ in panels for e in es})
    order = [target] + [c for c in order if c != target]
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / len(order)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(order, angles)}
    small = len(order) > 25
    fig, axes = plt.subplots(2, 3, figsize=(20, 14.5))
    for ax, (title, es, cut, mn, seconds) in zip(axes.flat, panels):
        for e in es:
            gone = e in cut
            a, c = split(e)
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=red if gone else grey,
                    linewidth=1.7 if gone else 1.1, alpha=0.85 if gone else 0.5, linestyle=(0, (4, 3)) if gone else "-")
        for col, (x, y) in pos.items():
            fill = red if col == target else "#FAC775" if col in mn else "white"
            ax.scatter([x], [y], s=150 if small else 360, color=fill, edgecolors="#5F5E5A", linewidths=0.7, zorder=2)
            ax.text(x * 1.17, y * 1.17, label(name, col), ha="center", va="center", fontsize=6.3 if small else 8.3)
        extra = f", {len(cut)} split" if cut else ""
        ax.set_title(f"{title}: {len(es)} edges{extra}\nMN target dataset: {len(mn)} columns; {seconds:.1f} s", fontsize=11)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=grey, lw=1.5, label="edge"),
                        Line2D([], [], color=red, lw=1.8, ls=(0, (4, 3)), label="triangle edge A-C split (explained by B)"),
                        Line2D([], [], marker="o", ls="", markerfacecolor=red, markeredgecolor="#5F5E5A", ms=10,
                               label=f"target ({label(name, target)})"),
                        Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=10,
                               label="in the target's MN dataset")],
               loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, fontsize=10.5, frameon=False)
    fig.suptitle(f"{name}: top row V0 (the jar) and the old V2 (old Python LIMA); bottom row the new V2 (on the port of the jar)",
                 x=0.02, y=0.995, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def part_b() -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    stored = [r for r in json.loads((STORED / "old.json").read_text()) if r["version"] in ("V0", "V1", "V2")]
    out, summary = [], []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        runs = []
        for r in run_versions(df, OLD_SEEDS):
            g = r["graph"]
            runs.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=sorted("-".join(sorted(e)) for e in g.edges),
                             groups=r["groups"], mn_target=target_dataset(r["groups"], target), splits=r["splits"], tests=r["tests"],
                             seconds=round(r["seconds"], 3), seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        for s in stored:
            if s["benchmark"] == name:
                v = {"V0": "V0 jar (stored)", "V1": "V1 old", "V2": "V2 old"}[s["version"]]
                runs.append(dict(benchmark=name, version=v, seed=s["seed"], edges=s["edges"], groups=s["groups"],
                                 mn_target=s["mn_target"], splits=s["splits"], tests=s["tests"], seconds=s["seconds"],
                                 seconds_extra=s.get("seconds_extra", 0.0)))
        get = lambda v, s: next(r for r in runs if r["version"] == v and r["seed"] == s)
        jar = get("V0", 42)
        assert set(jar["edges"]) == set(get("V0 jar (stored)", 0)["edges"]), f"{name}: V0 seed 42 differs from the stored jar"
        mn0 = set(jar["mn_target"])
        for r in runs:
            cut = {"-".join(sorted((t["a"], t["c"]))) for t in r["splits"]}
            mn = set(r["mn_target"])
            summary.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=len(r["edges"]), tests=len(r["tests"]),
                                splits=len(cut), split_edges=" ".join(sorted(cut)), groups=len(r["groups"]),
                                largest_group=max((len(g) for g in r["groups"]), default=0), mn_size=len(mn),
                                mn_overlap_jar=round(len(mn & mn0) / len(mn | mn0), 3), seconds=r["seconds"],
                                seconds_v2_step=r["seconds_extra"]))
        out += runs
        cut = lambda r: {"-".join(sorted((t["a"], t["c"]))) for t in r["splits"]}
        panels = [("V0 = the jar", jar["edges"], set(), jar["mn_target"], get("V0 jar (stored)", 0)["seconds"])]
        panels += [(f"V2 old, seed {s}", get("V2 old", s)["edges"], cut(get("V2 old", s)), get("V2 old", s)["mn_target"],
                    get("V2 old", s)["seconds"]) for s in (0, 1)]
        panels += [(f"V2 new, seed {s}", get("V2 new", s)["edges"], cut(get("V2 new", s)), get("V2 new", s)["mn_target"],
                    get("V2 new", s)["seconds"]) for s in OLD_SEEDS]
        plot(name, target, panels, OUT / "png" / f"{name}.png")
        print(f"B: {name} done", flush=True)
    (OUT / "old.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(summary).to_csv(OUT / "old.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", nargs="+", default=["A", "B"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if "A" in args.part:
        part_a()
    if "B" in args.part:
        part_b()
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
