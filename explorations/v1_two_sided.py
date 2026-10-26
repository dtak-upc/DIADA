"""
V1 = V0 (diada.core.LIMA, the Python port of the jar) with one change: a two-sided test.

V0 (the jar's test). For an edge X -> X + p and each q in X, compare p's rate with and without q:
    d = (meanLogOdds(p | X - q) - meanLogOdds(p | X)) / sqrt(var + var)
  d > 0 means adding q makes p rarer. The edge is sound (explored further) if every d >= 2, and it
  scores the smallest d; DIADA keeps the column pairs scoring > 4.
V1. The same with |d|: adding q may make p significantly rarer OR more common. The edge is sound if
  every |d| >= 2, and it scores the smallest |d|.

Only LIMA.judge is overridden; sampling, the search and the output are V0's. Nothing under
src/diada/ is modified.

The previous V1 (explorations/lima_two_sided.py) applied the same change to the old Python LIMA,
which has since been replaced by the port of the jar; its results are in results/v0_v3.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import pandas as pd

from diada.core.diada_tool import _label_columns, _relatedness_matrix_to_soundness_df
from diada.core.LIMA import DEFAULT_SEED, LIMA, Dataset


class TwoSidedLIMA(LIMA):
    def judge(self, distances: List[float]) -> Tuple[float, bool]:
        size = [abs(d) for d in distances]
        return min(size), all(s >= self.explore for s in size)


def run(df: pd.DataFrame, two_sided: bool = True, seed: int = DEFAULT_SEED, num_buckets: int = 10) -> dict:
    """V1 (or V0 with two_sided=False) on `df` as DIADA calls LIMA: label/bucket the columns, read
    them as the jar would, run, convert the matrix to DIADA's soundness triples. The time includes
    reading the labelled table."""
    labelled = _label_columns(df.copy(), num_buckets)
    start = time.perf_counter()
    dataset = Dataset.from_frame(labelled)
    lima = (TwoSidedLIMA if two_sided else LIMA)(dataset, seed=seed)
    matrix = lima.run()
    seconds = time.perf_counter() - start
    return dict(lima=lima, soundness=_relatedness_matrix_to_soundness_df(matrix, dataset.columns), seconds=seconds,
                **lima.stats)
