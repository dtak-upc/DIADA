"""
LIMA, the algorithm behind DIADA's soundness scores, in Python: a port of the algorithm inside
DIADA-0.8.jar that gives the jar's output exactly (see lima.py), plus the Apriori/FP-Growth
baselines of Experiment 1 (baselines.py) on the same machinery.

  lima.py         Dataset, LIMA (the algorithm), run_lima (CSV/DataFrame -> output matrix)
  java_random.py  java.util.SplittableRandom, the jar's pair sampler, bit for bit
  baselines.py    Apriori, FPGrowth

diada.core.diada_tool.invoke_diada(engine="lima") is where DIADA calls it.

This replaces the earlier port of the separate LIMA_py repository (slower, and its edges differed
from the jar's); that code is in the git history before this change.
"""

from .lima import DEFAULT_APPROX, DEFAULT_SEED, LIMA, Dataset, run_lima

__all__ = ["DEFAULT_APPROX", "DEFAULT_SEED", "LIMA", "Dataset", "run_lima"]
