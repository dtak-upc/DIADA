import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.util.nptypes import IndexLike


class TPDomainSet(DomainSet[np.int32]):
    """A DomainSet of tuple pairs: an (n, 2) int32 array of row-index pairs."""

    def __init__(self, pairs: npt.NDArray[np.int32]):
        self.pairs: npt.NDArray[np.int32] = pairs

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def get(self, indices: IndexLike) -> npt.NDArray[np.int32]:
        return self.pairs[indices]

    def getFull(self) -> npt.NDArray[np.int32]:
        return self.pairs  # already exactly this data, no copy needed
