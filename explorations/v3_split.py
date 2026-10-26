"""
V3 = V2 (explorations/v2_split.py, on the new base: the port of the jar + the two-sided test) +
deciding the open paths A-B-C (no A-C edge).

The same V3 as explorations/v3_open_paths.py (which ran on the old Python LIMA, now replaced); only
the import of V2 changed.

After V2, for every column B and every two of its neighbours A and C that are not linked (and
were not already split by V2, which includes the open paths V2's own splits created):

    Once B is known, is A related to C?
      yes  A and C relate jointly through B (as in sum_collider, interaction): keep together
      no   A and C each relate to B and to nothing else: A and C may not share a group

The test is V2's, unchanged (same sample of pairs, pairs with the same value of B, each value
separately combined with precision weights, LIMA's m and variance, confidence 4). Only the
reference changes: an open path has no A-C link, so "negligible" is measured against the weaker
link of the chain:
    A and C unrelated given B  if  |m| + 4 * sd(m)  <  0.2 * min(link A-B, link B-C)
"Unsure" means keep together, as in V2.

A pair A, C with several middle columns (A-B-C and A-D-C) may not share a group only if every
middle column shows them unrelated; one interaction is enough to keep them together.

V3 removes no edges (A-C has none); it adds the pairs it separates to V2's, and the groups are
built as in V2: the largest connected sets of columns with no separated pair.

Nothing under src/diada/ is modified.
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict

import networkx as nx

from v2_split import DEVS, EXPLAINED, N_PAIRS, effect_given, find_splits, PairSample, split_groups


def open_path_test(s: PairSample, a: str, b: str, c: str) -> dict:
    """Once B is known, is A related to C? (open path A-B-C, see the module docstring)"""
    chain = min(s.link(a, b), s.link(b, c))
    bound = (1 - EXPLAINED) * chain
    e = effect_given(s, a, b, c)
    return dict(a=a, b=b, c=c, weaker_link=round(chain, 4), bound=round(bound, 4), m_given_b=round(e["m"], 4),
                z_given_b=round(e["z"], 2), upper=round(e["upper"], 4),
                share_upper=round(e["upper"] / chain, 3) if chain > 0 else float("inf"), worst_pattern=e["pattern"],
                values_of_b=e["values"], unrelated=bool(e["upper"] < bound), interaction=bool(abs(e["z"]) > DEVS))


def find_separations(graph: nx.Graph, already: set, s: PairSample):
    """Test every open path of `graph` (the graph after V2) whose ends V2 did not already split;
    a pair is separated if every middle column shows its ends unrelated. Returns (separated
    pairs, every test)."""
    tests, by_pair = [], defaultdict(list)
    for b in sorted(graph.nodes):
        for a, c in itertools.combinations(sorted(graph.neighbors(b)), 2):
            if graph.has_edge(a, c) or frozenset((a, c)) in already:
                continue
            t = open_path_test(s, a, b, c)
            tests.append(t)
            by_pair[frozenset((a, c))].append(t)
    separated = [dict(a=sorted(k)[0], c=sorted(k)[1], middles=[t["b"] for t in ts],
                      share_upper=max(t["share_upper"] for t in ts))
                 for k, ts in by_pair.items() if all(t["unrelated"] for t in ts)]
    return separated, tests


def run_v2_v3(lima, graph: nx.Graph, n_pairs: int = N_PAIRS) -> dict:
    """V2 and V3 on top of a V1 run (its LIMA object and DIADA graph), on one sample of pairs."""
    start = time.perf_counter()
    s = PairSample(lima, n_pairs)
    g2, splits, tests = find_splits(graph, s)
    sep2 = {frozenset((t["a"], t["c"])) for t in splits}
    groups2 = split_groups(g2, sep2)
    mid = time.perf_counter()
    separated, open_tests = find_separations(g2, sep2, s)
    groups3 = split_groups(g2, sep2 | {frozenset((t["a"], t["c"])) for t in separated})
    end = time.perf_counter()
    return dict(v2=dict(graph=g2, splits=splits, tests=tests, groups=groups2, seconds=mid - start),
                v3=dict(graph=g2, splits=splits, tests=tests, separated=separated, open_tests=open_tests,
                        groups=groups3, seconds=end - start))
