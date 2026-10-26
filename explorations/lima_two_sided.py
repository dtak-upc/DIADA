"""
DIADA's Python LIMA with one change: a two-sided soundness test.

Original (diada.core.LIMA.LIMA.lattice.LIMALatticeState):
    an edge is sound if every comparison has z < -devs
    (adding the predicate makes the pattern significantly RARER)
Two-sided (this file):
    an edge is sound if every comparison has |z| > devs
    (adding the predicate makes the pattern significantly rarer OR more common)

Only two methods are overridden; nothing under src/diada/ is modified:
    _isSound     the test itself
    _minSoundZ   what relatedness() writes into the matrix: the original takes the most
                 negative z (written as -z); here the comparison with the largest |z|
                 (written as |z|), so the matrix stays positive and comparable
Everything else (sampling, scheduling, edge proposals, relatedness) is the original code.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from diada.core.diada_tool import _label_columns, _relatedness_matrix_to_soundness_df
from diada.core.LIMA.LIMA.LIMA import LIMA
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LIMALatticeState import LIMALatticeState
from diada.core.LIMA.util.stats.betaDist import nodeDist4


def comparisons(state, edge: Lattice.Edge) -> List[Tuple[int, float]]:
    """(removed column, z) for every comparison the original _isSound makes on `edge`."""
    parent, to1 = edge.fr(), edge.to()
    out = []
    for pg in parent.preds:
        sub = parent.fr(pg).fr()
        to2 = sub.to(edge.pg, edge.p).to()
        out.append((pg, nodeDist4(state.getDist(parent), state.getDist(to1), state.getDist(sub), state.getDist(to2))))
    return out


class TwoSidedLatticeState(LIMALatticeState):
    def __init__(self, lattice: Lattice, two_sided: bool = True) -> None:
        super().__init__(lattice)
        self.two_sided = two_sided

    def _isSound(self, edge, threshold):
        if not self.two_sided:
            return super()._isSound(edge, threshold)
        return all(abs(z) > threshold for _, z in comparisons(self, edge))

    def _minSoundZ(self, edge):
        if not self.two_sided:
            return super()._minSoundZ(edge)
        comps = comparisons(self, edge)
        return -max(abs(z) for _, z in comps) if comps else float("inf")


class TwoSidedLIMA(LIMA):
    def __init__(self, dataset, two_sided: bool = True, **kwargs):
        super().__init__(dataset, **kwargs)
        self.state = TwoSidedLatticeState(self.lattice, two_sided=two_sided)


def run(df: pd.DataFrame, version: str, seed: int, workdir, num_buckets: int = 10) -> dict:
    """Run "base" (the original LIMA) or "two_sided" on `df` exactly as DIADA's lima engine would:
    label/bucket the columns, write a CSV, run LIMA to completion, convert the matrix to
    DIADA's soundness triples. Returns the LIMA object, the soundness table and timing."""
    labelled = _label_columns(df.copy(), num_buckets)
    path = workdir / "input.csv"
    labelled.to_csv(path, index=False)
    np.random.seed(seed)
    start = time.perf_counter()
    lima = LIMA(str(path)) if version == "base" else TwoSidedLIMA(str(path))
    rounds = 0
    for i in range(100):
        if lima.step(i):
            break
        rounds = i + 1
    seconds = time.perf_counter() - start
    soundness = _relatedness_matrix_to_soundness_df(lima.relatedness(), labelled.columns.tolist())
    return dict(lima=lima, soundness=soundness, seconds=seconds, rounds=rounds,
                pairs=sum(lima.getStepSize(i) for i in range(rounds)), nodes=len(lima.state._dists))


def pair_z(lima) -> dict:
    """For every column pair, the most negative and the most positive z among the
    one-column -> two-column edges between them (from the final counts)."""
    st, root = lima.state, lima.lattice.getRoot()
    cols = [c.split("(")[0] for c in lima.dataset.columns]
    out = {}
    for x in range(len(cols)):
        for y in range(x + 1, len(cols)):
            zs = []
            for px in range(lima.lattice.pgs[x]):
                for py in range(lima.lattice.pgs[y]):
                    edge = root.to(x, px).to().to(y, py)
                    d = st.getDist(edge.to())
                    if d.a + d.b > 2:  # counted at least once
                        zs.append(comparisons(st, edge)[0][1])
            if zs:
                out[f"{cols[x]}-{cols[y]}"] = (min(zs), max(zs))
    return out
