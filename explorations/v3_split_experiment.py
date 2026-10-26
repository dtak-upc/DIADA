"""
V0-V3 on the new base (diada.core.LIMA, the Python port of the jar) vs the original V1-V3 (on the
old Python LIMA, stored in results/v0_v3).

  V0       diada.core.LIMA = the jar's output (seed 42 is the jar's seed)
  V1 new   V0 with the two-sided test (explorations/v1_two_sided.py)
  V2 new   V1 + splitting the triangles B explains (explorations/v2_split.py)
  V3 new   V2 + separating the open paths A-B-C whose ends are unrelated given B
           (explorations/v3_split.py); V2 and V3 use one test sample per run
  V1 old, V2 old, V3 old   stored results (seeds 0-4 synthetic, 0-1 old benchmarks)

Synthetic: the nine three-column benchmarks, scored on the groups against the expected ones.
Seeds 42 and 0-4. Old benchmarks: diamonds, bank, house_sales, bank_mn; seeds 42, 0, 1.

Run with:  python explorations/v3_split_experiment.py [--part A B]
Writes results/v3_new/{synthetic.csv, synthetic_summary.csv, old.json, old.csv, png/<name>.png}
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
from v2_split import target_dataset  # noqa: E402
from v3_split import run_v2_v3  # noqa: E402

OUT = paths.RESULTS_DIR / "v3_new"
STORED = paths.RESULTS_DIR / "v0_v3"
SYNTH_SEEDS, OLD_SEEDS = (42, 0, 1, 2, 3, 4), (42, 0, 1)
VERSION_ORDER = {"V0": 0, "V1 old": 1, "V1 new": 2, "V2 old": 3, "V2 new": 4, "V3 old": 5, "V3 new": 6}


def components(graph: nx.Graph) -> list:
    return sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda g: (-len(g), g))


def grouping_text(groups, columns) -> str:
    groups = list(groups) + [[c] for c in columns if not any(c in g for g in groups)]
    return " | ".join(sorted("-".join(sorted(g)) for g in groups))


def run_versions(df: pd.DataFrame, seeds) -> list:
    """V0 to V3 per seed: dict(version, seed, graph (LIMA's graph, before any split), groups,
    splits, separated, tests, open_tests, seconds, seconds_extra)"""
    out = []
    for seed in seeds:
        r0 = run(df, two_sided=False, seed=seed)
        g0 = build_soundness_graph(r0["soundness"], threshold=4)
        out.append(dict(version="V0", seed=seed, graph=g0, groups=components(g0), splits=[], separated=[], tests=[],
                        open_tests=[], seconds=r0["seconds"]))
        r1 = run(df, two_sided=True, seed=seed)
        g1 = build_soundness_graph(r1["soundness"], threshold=4)
        out.append(dict(version="V1 new", seed=seed, graph=g1, groups=components(g1), splits=[], separated=[], tests=[],
                        open_tests=[], seconds=r1["seconds"]))
        v = run_v2_v3(r1["lima"], g1)
        out.append(dict(version="V2 new", seed=seed, graph=g1, groups=v["v2"]["groups"], splits=v["v2"]["splits"], separated=[],
                        tests=v["v2"]["tests"], open_tests=[], seconds=r1["seconds"] + v["v2"]["seconds"],
                        seconds_extra=v["v2"]["seconds"]))
        out.append(dict(version="V3 new", seed=seed, graph=g1, groups=v["v3"]["groups"], splits=v["v3"]["splits"],
                        separated=v["v3"]["separated"], tests=v["v3"]["tests"], open_tests=v["v3"]["open_tests"],
                        seconds=r1["seconds"] + v["v3"]["seconds"], seconds_extra=v["v3"]["seconds"]))
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
                             separated=" ".join(f"{t['a']}-{t['c']}" for t in r["separated"]),
                             tests=" ".join(f"{t['a']}-{t['c']}|{t['b']}:{t['share_upper']:.2f}" for t in r["tests"]),
                             open_tests=" ".join(f"{t['a']}-{t['c']}|{t['b']}:{t['share_upper']:.2f}"
                                                 f"{'(interaction)' if t['interaction'] else ''}" for t in r["open_tests"]),
                             grouping=got, expected=want, ok=None if want == "n/a" else got == want,
                             seconds=round(r["seconds"], 3), seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        for _, s in stored[(stored.benchmark == name) & stored.version.isin(["V1", "V2", "V3"])].iterrows():
            text = lambda v: v if isinstance(v, str) else ""
            rows.append(dict(benchmark=name, version=f"{s.version} old", seed=int(s.seed), edges=s.edges, graph_ok=bool(s.graph_ok),
                             splits=text(s.splits), separated=text(s.separated), tests=text(s.tests), open_tests=text(s.open_tests),
                             grouping=s.grouping, expected=want, ok=None if want == "n/a" else s.grouping == want,
                             seconds=s.seconds, seconds_extra=s.seconds_extra))
        print(f"A: {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "synthetic.csv", index=False)
    summary = out.groupby(["benchmark", "version"], sort=False).agg(
        runs=("seed", "size"), grouping_ok=("ok", lambda s: s.sum() if s.notna().all() else np.nan),
        groupings=("grouping", lambda s: " / ".join(f"{g} ({n})" for g, n in s.value_counts().items())),
        extra_seconds=("seconds_extra", "mean")).reset_index()
    summary["b"] = summary.benchmark.map(list(EXPECTED).index)
    summary["o"] = summary.version.map(VERSION_ORDER)
    summary.sort_values(["b", "o"]).drop(columns=["b", "o"]).to_csv(OUT / "synthetic_summary.csv", index=False)


def plot(name: str, target: str, panels: list, path: Path) -> None:
    """panels: (title, edges as "A-B" strings, split edges, separated pairs, MN target dataset, seconds)"""
    grey, red, purple = "#888780", "#E24B4A", "#7F77DD"
    split = lambda e: tuple(e.split("-", 1))
    nodes = sorted({c for p in panels for e in p[1] for c in split(e)} | {target})
    order = [c for c in BANK_ORDER if c in nodes] if name == "bank" else spread_order(nodes, {split(e) for p in panels for e in p[1]})
    order = [target] + [c for c in order if c != target]
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / len(order)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(order, angles)}
    small = len(order) > 25
    fig, axes = plt.subplots(2, 3, figsize=(20, 14.5))
    for ax, (title, es, cut, sep, mn, seconds) in zip(axes.flat, panels):
        for e in es:
            gone = e in cut
            a, c = split(e)
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=red if gone else grey,
                    linewidth=1.7 if gone else 1.1, alpha=0.85 if gone else 0.5, linestyle=(0, (4, 3)) if gone else "-")
        for e in sep:
            a, c = split(e)
            ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], zorder=1, color=purple, linewidth=1.6, alpha=0.9,
                    linestyle=(0, (1, 2)))
        for col, (x, y) in pos.items():
            fill = red if col == target else "#FAC775" if col in mn else "white"
            ax.scatter([x], [y], s=150 if small else 360, color=fill, edgecolors="#5F5E5A", linewidths=0.7, zorder=2)
            ax.text(x * 1.17, y * 1.17, label(name, col), ha="center", va="center", fontsize=6.3 if small else 8.3)
        extra = (f", {len(cut)} split" if cut else "") + (f", {len(sep)} open paths cut" if sep else "")
        ax.set_title(f"{title}: {len(es)} edges{extra}\nMN target dataset: {len(mn)} columns; {seconds:.1f} s", fontsize=11)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=grey, lw=1.5, label="edge"),
                        Line2D([], [], color=red, lw=1.8, ls=(0, (4, 3)), label="V2: triangle edge A-C split (explained by B)"),
                        Line2D([], [], color=purple, lw=1.8, ls=(0, (1, 2)), label="V3: open path A-B-C cut (A, C unrelated given B)"),
                        Line2D([], [], marker="o", ls="", markerfacecolor=red, markeredgecolor="#5F5E5A", ms=10,
                               label=f"target ({label(name, target)})"),
                        Line2D([], [], marker="o", ls="", markerfacecolor="#FAC775", markeredgecolor="#5F5E5A", ms=10,
                               label="in the target's MN dataset")],
               loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, fontsize=10.5, frameon=False)
    fig.suptitle(f"{name}: top row V0 (the jar) and the old V3 (old Python LIMA); bottom row the new V3 (on the port of the jar)",
                 x=0.02, y=0.995, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def part_b() -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    stored = [r for r in json.loads((STORED / "old.json").read_text()) if r["version"] in ("V0", "V1", "V2", "V3")]
    out, summary = [], []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        runs = []
        for r in run_versions(df, OLD_SEEDS):
            runs.append(dict(benchmark=name, version=r["version"], seed=r["seed"],
                             edges=sorted("-".join(sorted(e)) for e in r["graph"].edges), groups=r["groups"],
                             mn_target=target_dataset(r["groups"], target), splits=r["splits"], separated=r["separated"],
                             tests=r["tests"], open_tests=r["open_tests"], seconds=round(r["seconds"], 3),
                             seconds_extra=round(r.get("seconds_extra", 0.0), 3)))
        for s in stored:
            if s["benchmark"] == name:
                v = {"V0": "V0 jar (stored)", "V1": "V1 old", "V2": "V2 old", "V3": "V3 old"}[s["version"]]
                runs.append(dict(benchmark=name, version=v, seed=s["seed"], edges=s["edges"], groups=s["groups"],
                                 mn_target=s["mn_target"], splits=s["splits"], separated=s["separated"], tests=s["tests"],
                                 open_tests=s["open_tests"], seconds=s["seconds"], seconds_extra=s.get("seconds_extra", 0.0)))
        get = lambda v, s: next(r for r in runs if r["version"] == v and r["seed"] == s)
        jar = get("V0", 42)
        assert set(jar["edges"]) == set(get("V0 jar (stored)", 0)["edges"]), f"{name}: V0 seed 42 differs from the stored jar"
        mn0 = set(jar["mn_target"])
        cut = lambda r: {"-".join(sorted((t["a"], t["c"]))) for t in r["splits"]}
        sep = lambda r: {"-".join(sorted((t["a"], t["c"]))) for t in r["separated"]}
        for r in runs:
            mn = set(r["mn_target"])
            summary.append(dict(benchmark=name, version=r["version"], seed=r["seed"], edges=len(r["edges"]), tests=len(r["tests"]),
                                splits=len(cut(r)), open_tests=len(r["open_tests"]),
                                interactions=sum(t["interaction"] for t in r["open_tests"]), separated=len(sep(r)),
                                separated_pairs=" ".join(sorted(sep(r))), groups=len(r["groups"]),
                                largest_group=max((len(g) for g in r["groups"]), default=0), mn_size=len(mn),
                                mn_overlap_jar=round(len(mn & mn0) / len(mn | mn0), 3), seconds=r["seconds"],
                                seconds_extra=r["seconds_extra"]))
        out += runs
        panels = [("V0 = the jar", jar["edges"], set(), set(), jar["mn_target"], get("V0 jar (stored)", 0)["seconds"])]
        panels += [(f"V3 old, seed {s}", get("V3 old", s)["edges"], cut(get("V3 old", s)), sep(get("V3 old", s)),
                    get("V3 old", s)["mn_target"], get("V3 old", s)["seconds"]) for s in (0, 1)]
        panels += [(f"V3 new, seed {s}", get("V3 new", s)["edges"], cut(get("V3 new", s)), sep(get("V3 new", s)),
                    get("V3 new", s)["mn_target"], get("V3 new", s)["seconds"]) for s in OLD_SEEDS]
        plot(name, target, panels, OUT / "png" / f"{name}.png")
        print(f"B: {name} done", flush=True)
    (OUT / "old.json").write_text(json.dumps(out, indent=1))
    df = pd.DataFrame(summary)
    df["o"] = df.version.map(lambda v: VERSION_ORDER.get(v, -1))
    df.sort_values(["benchmark", "o", "seed"], key=lambda c: c.map(list(OLD).index) if c.name == "benchmark" else c) \
        .drop(columns="o").to_csv(OUT / "old.csv", index=False)


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
