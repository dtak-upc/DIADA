import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.random.RandZ2 import RandZ2


class TPSet:
    """A set of tuple pairs: an (n, 2) int32 array of row-index pairs."""

    def __init__(self, pairs: npt.NDArray[np.int32]):
        self.pairs: npt.NDArray[np.int32] = pairs

    def __len__(self) -> int:
        return self.pairs.shape[0]


def generate_tpset(n: int, range_: int, rz2: RandZ2 = None) -> TPSet:
    """Generate a TPSet by calling RandZ2.uniformGrid(n, range_)."""
    rz2 = rz2 if rz2 is not None else RandZ2()
    return TPSet(rz2.uniformGrid(n, range_))
