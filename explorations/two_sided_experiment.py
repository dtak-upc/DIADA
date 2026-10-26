"""
V0 vs. V1 (the two-sided test, explorations/lima_two_sided.py) vs. V2 / V2.1 (mediation
pruning, explorations/lima_mediation.py: V2 = V1 + pruning, V2.1 = one-sided LIMA + pruning,
V2A = V2 with the pruning test run within each value of the mediator, V3 = V2A + split
grouping, explorations/split_grouping.py, where groups may overlap; V2.exact = the same split
with exact counts over all row pairs instead of LIMA's sample, explorations/exact_counts.py),
where V0 is either
    --v0 jar   DIADA's default engine, DIADA-0.8.jar via invoke_diada(engine="jar")
               (deterministic, so it runs once per benchmark); results in results/jar_vs_two_sided/
    --v0 base  the original Python LIMA port; results in results/two_sided/

Part A, synthetic: every three-variable benchmark in explorations/synthetic_triads.py,
    5 seeds each. Per run: the graph DIADA builds (pairs with soundness > 4, via
    dataset_builder.build_soundness_graph), the matrix value per pair, per pair the most
    negative and most positive z among its one-column -> two-column edges, and a score
    against the expected results: is the graph right (edges missing / extra, against
    EXPECTED_EDGES), and is the grouping DIADA would build from it (its connected
    components; for V3 its split groups) the expected one.

Part B, original benchmarks: diamonds, bank, house_sales (data/base_datasets) and bank
    with injected noise (data/noise_mn/bank_mn_noise.csv), 2 seeds each. Per run: the
    graph, DIADA's UN selection (columns with an edge) and MN target cluster (for V3, the
    union of the split groups containing the target), the groups, the PNG
    drawn by dataset_builder.visualize_clusters, mutual information of every edge on
    the bucketed data, and for bank MN the injected columns that got linked.

Run with:  python explorations/two_sided_experiment.py [--v0 jar|base] [--part A B] [--versions ...] [--skip-v0]
Writes <out>/{synthetic_runs.csv, old_runs.json, png/*.png}; runs of versions not run this time
are kept from the existing files, so a single version can be added or re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from diada import paths
from diada.core.dataset_builder import build_soundness_graph, visualize_clusters
from diada.core.diada_tool import _label_columns, invoke_diada

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import run_exact  # noqa: E402
from lima_mediation import run_mediation  # noqa: E402
from lima_two_sided import pair_z, run  # noqa: E402
from synthetic_triads import EXPECTED, EXPECTED_EDGES  # noqa: E402

OUT_DIRS = {"base": paths.RESULTS_DIR / "two_sided", "jar": paths.RESULTS_DIR / "jar_vs_two_sided"}
OUT = OUT_DIRS["base"]  # set in main() from --v0
SYNTHETIC = list(EXPECTED)
LABEL = {"base": "V0 base", "jar": "V0 jar", "two_sided": "V1 two-sided", "v2": "V2 two-sided + mediation",
         "v2_1": "V2.1 one-sided + mediation", "v2a": "V2A two-sided + stratified mediation", "v3": "V3 V2A + split grouping", "v2_exact": "V2.exact"}
# The grouping layer. "A-B | B-C" (cut at B, B in both groups) cannot come out of connected
# components, only out of the split grouping (V3); xor is not detectable by pairwise methods.
EXPECTED_GROUPING = {"independent": "A | B | C", "pair_isolated": "A-B | C", "chain": "A-B | B-C",
                     "common_cause": "A-B | B-C", "hub_join": "A-B | B-C", "direct_triangle": "A-B-C",
                     "sum_collider": "A-B-C", "xor": "n/a", "interaction": "A-B-C"}
OLD = {"diamonds": ("base_datasets/diamonds.csv", "price"), "bank": ("base_datasets/bank.csv", "Class"),
       "house_sales": ("base_datasets/house_sales.csv", "price"), "bank_mn": ("noise_mn/bank_mn_noise.csv", "Class")}
VERSIONS = ["base", "two_sided"]  # set in main() from --v0


def run_version(df: pd.DataFrame, version: str, seed: int, workdir: Path) -> dict:
    """Run one version. The jar goes through DIADA's own invoke_diada(engine="jar"), exactly as
    DIADA calls it; it is deterministic and has no LIMA object to inspect."""
    if version == "v2_exact":
        return run_exact(df, seed=seed)
    if version in ("v2", "v2_1", "v2a", "v3"):
        return run_mediation(df, two_sided=version != "v2_1", seed=seed, workdir=workdir, stratified=version in ("v2a", "v3"))
    if version != "jar":
        return run(df, version, seed, workdir)
    start = time.perf_counter()
    soundness = invoke_diada(df, 10, engine="jar", verbose=False)
    return dict(lima=None, soundness=soundness, seconds=time.perf_counter() - start, rounds=None, pairs=None, nodes=None)


def seeds_for(version: str, seeds) -> list:
    return [0] if version == "jar" else list(seeds)


def directed(soundness: pd.DataFrame, a: str, b: str) -> float:
    """The engine's own score for column a -> column b (0 if not reported)."""
    hit = soundness[(soundness.column_1 == a) & (soundness.column_2 == b)]["soundness"]
    return round(float(hit.max()), 2) if len(hit) else 0.0


def grouping(edges: set, groups=None) -> str:
    """The groups of A, B, C, e.g. "A-B | C", "A-B-C" or "A-B | B-C": the connected components
    of the graph, or the given split groups (columns in no group as singletons)."""
    if groups is None:
        g = nx.Graph()
        g.add_nodes_from("ABC")
        g.add_edges_from(tuple(e.split("-")) for e in edges)
        groups = nx.connected_components(g)
    else:
        groups = list(groups) + [[c] for c in "ABC" if not any(c in grp for grp in groups)]
    return " | ".join(sorted("-".join(sorted(c)) for c in groups))


def score(name: str, edges: set, groups=None) -> dict:
    """graph_ok: the pairs related on their own (EXPECTED_EDGES). final_ok: the graph the
    mediation step should leave, i.e. the same but without A-C where A-C is fully explained
    by B (the "split" benchmarks)."""
    must, may = EXPECTED_EDGES[name]
    missing, extra = must - edges, edges - must - may
    split = EXPECTED_GROUPING[name] == "A-B | B-C"
    f_must, f_may = (must - {"A-C"}, may - {"A-C"}) if split else (must, may)
    f_missing, f_extra = f_must - edges, edges - f_must - f_may
    want = EXPECTED_GROUPING[name]
    got = grouping(edges, groups)
    return dict(graph_ok=not missing and not extra, missing=" ".join(sorted(missing)), extra=" ".join(sorted(extra)),
                final_ok=not f_missing and not f_extra, final_missing=" ".join(sorted(f_missing)),
                final_extra=" ".join(sorted(f_extra)),
                grouping=got, grouping_ok=None if want == "n/a" else got == want)


def edges_of(soundness: pd.DataFrame) -> dict:
    g = build_soundness_graph(soundness, threshold=4)
    return {"-".join(sorted((a, b))): round(d["weight"], 2) for a, b, d in g.edges(data=True)}


def mutual_info_table(df: pd.DataFrame):
    """Mutual information (bits, Miller-Madow corrected) between bucketed columns, on demand."""
    lab = _label_columns(df.copy(), 10)
    codes = {c: pd.factorize(lab.iloc[:, i].astype(str))[0] for i, c in enumerate(df.columns)}

    def h(*cs):
        j = cs[0].astype(np.int64)
        for c in cs[1:]:
            j = j * (int(c.max()) + 1) + c
        cnt = np.unique(j, return_counts=True)[1]
        p = cnt / cnt.sum()
        return (-(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * cnt.sum())) / np.log(2)

    return lambda a, b: round(h(codes[a]) + h(codes[b]) - h(codes[a], codes[b]), 4)


def part_a() -> None:
    rows = []
    for name in SYNTHETIC:
        df = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / f"{name}.csv")
        for version in VERSIONS:
            for seed in seeds_for(version, range(5)):
                with tempfile.TemporaryDirectory() as d:
                    r = run_version(df, version, seed, Path(d))
                e = edges_of(r["soundness"])
                z = pair_z(r["lima"]) if r["lima"] is not None else {}
                row = dict(benchmark=name, version=LABEL[version], seed=seed, seconds=round(r["seconds"], 4), rounds=r["rounds"],
                           pairs=r["pairs"], nodes=r["nodes"], edges=" ".join(sorted(e)) or "none",
                           **score(name, set(e), r["groups"] if version in ("v3", "v2_exact") else None),
                           pruned=" ".join(f"{t['a']}-{t['c']}|{t['mediator']}" for t in r.get("pruned", [])),
                           conditional_evidence=" ".join(f"{t['a']}-{t['c']}|{t['mediator']}:{t.get('evidence', t.get('left'))}" for t in r.get("tested", [])),
                           seconds_prune=round(r.get("seconds_prune", 0.0), 4))
                for p in ("A-B", "B-C", "A-C"):
                    row[f"{p} value"] = e.get(p, 0.0)
                    row[f"{p} z-"], row[f"{p} z+"] = (round(z[p][0], 2), round(z[p][1], 2)) if p in z else (None, None)
                    x, y = p.split("-")
                    row[f"{p} score {x}->{y}"], row[f"{p} score {y}->{x}"] = directed(r["soundness"], x, y), directed(r["soundness"], y, x)
                rows.append(row)
        print(f"A: {name} done", flush=True)
    new = pd.DataFrame(rows)
    path = OUT / "synthetic_runs.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.version.isin(new.version.unique())], new], ignore_index=True)
    new.to_csv(path, index=False)


def kind(col: str) -> str:
    if col.startswith("noise_"):
        return "noise"
    if "_shuffled_" in col:
        return "shuffled"
    if col.startswith("spurious_"):
        return "spurious"
    return "original"


def part_b() -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    out = []
    for name, (rel, target) in OLD.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        mi = mutual_info_table(df)
        for version in VERSIONS:
            for seed in seeds_for(version, (0, 1)):
                with tempfile.TemporaryDirectory() as d:
                    r = run_version(df, version, seed, Path(d))
                e = edges_of(r["soundness"])
                g = nx.Graph()
                g.add_nodes_from(df.columns)
                g.add_edges_from(tuple(k.split("-", 1)) for k in e)
                png = OUT / "png" / f"{name}_{version}_seed{seed}.png"
                visualize_clusters(r["soundness"], target, output_path=str(png), threshold=4)
                groups = r["groups"] if version in ("v3", "v2_exact") else sorted((sorted(c) for c in nx.connected_components(g) if len(c) > 1),
                                                                    key=lambda c: (-len(c), c))
                mn_target = (sorted(set().union(*(c for c in groups if target in c))) if any(target in c for c in groups) else [target])                     if version in ("v3", "v2_exact") else sorted(nx.node_connected_component(g, target))
                out.append(dict(benchmark=name, version=LABEL[version], seed=seed, cols=df.shape[1], rows=len(df),
                                seconds=round(r["seconds"], 3), rounds=r["rounds"], pairs=r["pairs"], nodes=r["nodes"],
                                edges={k: dict(value=v, mi=mi(*k.split("-", 1))) for k, v in e.items()},
                                un_kept=sorted(n for n in g if g.degree(n) > 0),
                                mn_target_cluster=mn_target, groups=groups,
                                injected_linked=sorted(n for n in g if g.degree(n) > 0 and kind(n) != "original"),
                                pruned=r.get("pruned", []), triangles_tested=len(r.get("tested", [])),
                                seconds_prune=round(r.get("seconds_prune", 0.0), 3),
                                png=str(png.relative_to(paths.REPO_ROOT))))
                print(f"B: {name} {version} seed {seed}: {len(e)} edges, {r['seconds']:.1f}s", flush=True)
    path = OUT / "old_runs.json"
    if path.exists():
        done = {r["version"] for r in out}
        out = [r for r in json.loads(path.read_text()) if r["version"] not in done] + out
    path.write_text(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", nargs="+", default=["A", "B"])
    parser.add_argument("--v0", choices=["jar", "base"], default="jar")
    parser.add_argument("--versions", nargs="+", default=["two_sided", "v2", "v2_1", "v2a", "v3", "v2_exact"],
                        help="versions to compare against V0 (two_sided = V1, v2 = V2, v2_1 = V2.1, v2a = V2A, v3 = V3, "
                             "v2_exact = V2.exact)")
    parser.add_argument("--skip-v0", action="store_true", help="do not re-run V0 (keep its existing results)")
    args = parser.parse_args()
    global OUT, VERSIONS
    OUT, VERSIONS = OUT_DIRS[args.v0], ([] if args.skip_v0 else [args.v0]) + args.versions
    OUT.mkdir(parents=True, exist_ok=True)
    if "A" in args.part:
        part_a()
    if "B" in args.part:
        part_b()


if __name__ == "__main__":
    main()
