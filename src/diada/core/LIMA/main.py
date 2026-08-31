"""
The entry point of the LIMA port -- moved here verbatim from LIMA_py/main.py
(the sibling repo this was ported from), only the imports were rewritten to
the new package location. See diada.core.diada_tool.invoke_diada for where
this is actually called from within DIADA.
"""

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.LIMA.LIMA import LIMA
from diada.core.LIMA.LIMA.sampler.LIMASampler import LIMASampler
from diada.core.LIMA.LIMA.scheduler.LIMAScheduler import LIMAScheduler


#Safety bound, never reached.
MAX_ITERATIONS = 100


def runLima(dataset_path: str, approx: float) -> npt.NDArray[np.float64]:
    """Run LIMA to completion on `dataset_path` with the given approximation
    factor, then return its attribute relatedness matrix."""
    lima = LIMA(
        dataset_path,
        scheduler=LIMAScheduler(),
        sampler=LIMASampler(),
        approx=approx,
    )
    for i in range(MAX_ITERATIONS):
        if lima.step(i):
            break
    return lima.relatedness()

