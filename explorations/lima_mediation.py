"""
V2 / V2.1: remove "false links" from LIMA's graph, i.e. edges A-C that are fully explained
by a third column B.

    V2    two-sided LIMA (explorations/lima_two_sided.py) + mediation pruning (two-sided)
    V2.1  original one-sided LIMA + mediation pruning (one-sided)
    V2A   V2 with the test run within each value of B (stratified, see below)
    V3    V2A + split grouping (explorations/split_grouping.py): a triangle A-B-C whose A-C is
          explained by B is cut at B into {A, B} and {B, C}; everything else stays together

After LIMA finishes, for every triangle A-B, B-C, A-C in DIADA's graph (soundness > 4), the
edge A-C is tested for mediation by B: does C still tell anything about A once B is known?

The test is LIMA's own statistic (betaDist.nodeDist4) applied to three-column patterns:
for the edge that adds C (with predicate pc) to the node {A pa, B=}, the comparison that
removes A,

    m = [log p(A pa, B=, C pc) - log p(A pa, B=)] - [log p(B=, C pc) - log p(B=)]

i.e. "is C pc more or less likely among pairs that agree on B AND have A pa, than among
pairs that agree on B?". The same four terms give the reverse question (does A tell about
C beyond B), so one test per pattern covers both directions.

  - Only patterns where the two rows AGREE on B: that is what "given B" means for pairs.
    Pairs that disagree on B pool many different values of B, among which A and C can
    still look related (observed earlier on hub_join).
  - The four patterns (A =/!=, C =/!=) are tested; A-C is mediated by B if none of them is
    sound (V2: every |z| <= devs; V2.1: no z < -devs).
  - The counts come from a fresh sample of row pairs drawn with LIMA's own pair sampler
    (TPDomain) and predicates, so every triangle is tested, not only the patterns LIMA's
    search happened to count.
  - Candidates are pruned from the weakest conditional evidence up, and only if their
    triangle still exists at that moment, so two edges cannot remove each other.

Stratified (V2A): "the rows agree on B" pools pairs with different values of B (both rows
B = yes, both rows B = no, ...). When those groups have different baselines for C and A's
values are spread unevenly over them, pooling can cancel a dependence that is present
within every value of B (Simpson's paradox; bank pdays-month given housing_loan). V2A runs
the same four-pattern test separately on the pairs whose two rows both have B = v, for every
value v of B, and takes the strongest evidence over all values: A-C is mediated only if no
value of B shows a sound dependence. The value of B for a pair is read from LIMA's own
category codes (UnorderedPG._codes) at the pair's first row.

Nothing under src/diada/ is modified.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from diada.core.dataset_builder import build_soundness_graph
from diada.core.LIMA.util.stats.betaDist import BetaDist, nodeDist4

from lima_two_sided import run
from split_grouping import split_groups

N_PAIRS = 1 << 17  # LIMA's largest per-round sample size


def _dist(a: int, n: int) -> BetaDist:
    d = BetaDist.empty()
    d.add(int(a), int(n - a))
    return d


def conditional_evidence(agree: Dict[str, np.ndarray], a: str, b: str, c: str, two_sided: bool,
                         b_value: Optional[np.ndarray] = None, detail: bool = False):
    """Strongest evidence, over the patterns where the rows agree on b, that a and c are still
    related once b is known. Two-sided: the largest |z|; one-sided: the largest -z ("rarer").
    With b_value (the value of b for each pair), the test runs separately within each value of
    b and the strongest evidence over the values is returned; with detail, also a list of
    (value, pairs, evidence) per value."""
    n = len(agree[b])
    groups = [(None, agree[b])] if b_value is None else         [(v, agree[b] & (b_value == v)) for v in np.unique(b_value[agree[b]])]
    best, per_value = 0.0, []
    for v, on_b in groups:
        here = 0.0
        for pa, pc in itertools.product((True, False), repeat=2):
            a_side = agree[a] if pa else ~agree[a]
            c_side = agree[c] if pc else ~agree[c]
            from1, to1 = on_b & a_side, on_b & a_side & c_side
            to2 = on_b & c_side
            z = nodeDist4(_dist(from1.sum(), n), _dist(to1.sum(), n), _dist(on_b.sum(), n), _dist(to2.sum(), n))
            here = max(here, abs(z) if two_sided else -z)
        per_value.append((v, int(on_b.sum()), here))
        best = max(best, here)
    return (best, per_value) if detail else best


def mediation_prune(lima, soundness: pd.DataFrame, two_sided: bool, n_pairs: int = N_PAIRS, devs: float = 4.0,
                    stratified: bool = False):
    """Remove edges A-C of DIADA's graph that are fully explained by a third column B
    (stratified: tested within each value of B). Returns (pruned soundness table, log of
    pruned edges, log of every triangle tested, split groups of the pruned graph)."""
    graph = build_soundness_graph(soundness, threshold=devs)
    names = [c.split("(")[0] for c in lima.dataset.columns]
    pg_of = {n: lima.predicate_groups[i] for i, n in enumerate(names)}

    domain_set = lima.domain.makeDomainSet(n_pairs)
    agree = {n: pg_of[n].pgEval(domain_set)[0].astype(bool) for n in graph.nodes}
    first_row = domain_set.getFull()[:, 0]
    value = {n: pg_of[n]._codes[first_row] for n in graph.nodes} if stratified else {}

    tested = []
    for a, c in graph.edges:
        for b in nx.common_neighbors(graph, a, c):
            tested.append(dict(a=a, c=c, mediator=b, evidence=round(
                conditional_evidence(agree, a, b, c, two_sided, value.get(b)), 2)))

    pruned: List[dict] = []
    g = graph.copy()
    for t in sorted(tested, key=lambda t: t["evidence"]):
        if t["evidence"] > devs:
            break
        a, b, c = t["a"], t["mediator"], t["c"]
        if g.has_edge(a, c) and g.has_edge(a, b) and g.has_edge(b, c):
            g.remove_edge(a, c)
            pruned.append(t)

    gone = {frozenset((t["a"], t["c"])) for t in pruned}
    keep = np.array([frozenset((x, y)) not in gone for x, y in zip(soundness.column_1, soundness.column_2)], dtype=bool)
    return soundness.loc[keep].reset_index(drop=True), pruned, tested, split_groups(g, gone)


def run_mediation(df: pd.DataFrame, two_sided: bool, seed: int, workdir: Path, n_pairs: int = N_PAIRS,
                  stratified: bool = False) -> dict:
    """V2 (two_sided=True), V2.1 (two_sided=False) or V2A (two_sided=True, stratified=True):
    LIMA run, then mediation pruning."""
    r = run(df, "two_sided" if two_sided else "base", seed, workdir)
    start = time.perf_counter()
    pruned_soundness, pruned, tested, groups = mediation_prune(r["lima"], r["soundness"], two_sided, n_pairs, stratified=stratified)
    r.update(soundness=pruned_soundness, pruned=pruned, tested=tested, groups=groups,
             seconds_lima=r["seconds"], seconds_prune=time.perf_counter() - start)
    r["seconds"] = r["seconds_lima"] + r["seconds_prune"]
    return r
