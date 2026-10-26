"""
V0 (DIADA-0.8.jar) vs V1 (Python LIMA, two-sided test) vs V2 (V1 + splitting a triangle A-B-C
into {A, B} and {B, C} when B explains A-C; the "explained" test runs within each value of B;
stored in old_runs.json as "V3 V2A + split grouping") on the old benchmarks, diagnosed with
predicate computations only (explorations/predicate_diagnostics.py), no mutual information.

  edge_diffs.csv   every edge found by only one of V0 / V1 (seed 0): V1's own most negative
                   and most positive z for that pair (from its LIMA run), the fewest pairs
                   LIMA counted for any of the pair's four two-column nodes, the exact
                   correlation phi between "agree on x" and "agree on y" over all row pairs,
                   and the z a sample of 2^17 pairs would give
  v2_splits.csv    every triangle V2 split (both seeds): the evidence V2's sample gave, and the
                   exact phi(A, C | B) over all pairs (agreement correlation within each value
                   of B, averaged), next to phi(A, C), and their ratio (how much of the A-C
                   agreement correlation is left once B is known), with a verdict calibrated on
                   the synthetic triangles: ratio <= 0.2 explained (true mediation gives
                   0.03-0.10), >= 0.5 not explained, between: unclear

Run with:  python explorations/three_versions_report.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from lima_two_sided import pair_z, run  # noqa: E402
from predicate_diagnostics import bucket_codes, conditional_phi, pairwise, pairwise_phi  # noqa: E402
from two_sided_plots import SOURCES, label  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided"
V0, V1, V2 = "V0 jar", "V1 two-sided", "V3 V2A + split grouping"


def verdict(left: float) -> str:
    return "explained" if left <= 0.2 else "not explained" if left >= 0.5 else "unclear"


def fewest_pairs(lima, x: str, y: str) -> int:
    """Fewest pairs LIMA counted for any of the four nodes {x pattern, y pattern}."""
    cols = [c.split("(")[0] for c in lima.dataset.columns]
    i, j = cols.index(x), cols.index(y)
    root, st = lima.lattice.getRoot(), lima.state
    return min(int(st.getDist(root.to(i, px).to().to(j, py).to()).a + st.getDist(root.to(i, px).to().to(j, py).to()).b - 2)
               for px in range(lima.lattice.pgs[i]) for py in range(lima.lattice.pgs[j]))


def main() -> None:
    runs = {(r["benchmark"], r["version"], r["seed"]): r for r in json.loads((OUT / "old_runs.json").read_text())}
    diffs, splits = [], []
    for name, (rel, target) in SOURCES.items():
        df = pd.read_csv(paths.DATA_DIR / rel)
        codes = bucket_codes(df)
        with tempfile.TemporaryDirectory() as d:
            lima = run(df, "two_sided", 0, Path(d))["lima"]
        z = pair_z(lima)
        e0, e1 = set(runs[(name, V0, 0)]["edges"]), set(runs[(name, V1, 0)]["edges"])
        for k in sorted(e0 ^ e1):
            x, y = k.split("-", 1)
            zk = z.get(k) or z.get(f"{y}-{x}") or (float("nan"), float("nan"))
            ez = max(abs(e[3]) for e in pairwise(codes, x, y))
            diffs.append(dict(benchmark=name, edge=f"{label(name, x)}-{label(name, y)}", found_by="V0 only" if k in e0 else "V1 only",
                              V1_z_min=round(zk[0], 2), V1_z_max=round(zk[1], 2), V1_fewest_pairs=fewest_pairs(lima, x, y),
                              exact_phi=round(pairwise_phi(codes, x, y), 4), expected_z_at_2_17=round(ez, 1)))
        for seed in (0, 1):
            for t in runs[(name, V2, seed)]["pruned"]:
                a, b, c = t["a"], t["mediator"], t["c"]
                s = conditional_phi(codes, a, b, c)
                splits.append(dict(benchmark=name, seed=seed, edge=f"{label(name, a)}-{label(name, c)}", split_at=label(name, b),
                                   sampled_evidence=t["evidence"], phi_AC=s["phi_overall"], phi_AC_given_B=s["phi_given_b"],
                                   largest_phi_in_a_value=s["largest_phi_v"],
                                   left=round(s["phi_given_b"] / abs(s["phi_overall"]), 2),
                                   verdict=verdict(s["phi_given_b"] / abs(s["phi_overall"]))))
        print(f"{name} done", flush=True)
    d, s = pd.DataFrame(diffs), pd.DataFrame(splits)
    d.to_csv(OUT / "edge_diffs.csv", index=False)
    s.to_csv(OUT / "v2_splits.csv", index=False)
    print(d.to_string(index=False))
    print(s.to_string(index=False))
    print(pd.crosstab(s.benchmark, s.verdict))


if __name__ == "__main__":
    main()
