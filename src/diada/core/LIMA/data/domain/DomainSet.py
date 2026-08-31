from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import IndexLike
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet

T = TypeVar("T", bound=np.generic)


class DomainSet(ABC, Generic[T]):
    """Abstract indexable set of values: an instantiated subspace of a Domain."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements in this set."""
        raise NotImplementedError

    @abstractmethod
    def get(self, indices: IndexLike) -> npt.NDArray[T]:
        """Return the value(s) at the given position(s)."""
        raise NotImplementedError

    @abstractmethod
    def getFull(self) -> npt.NDArray[T]:
        """Return every value in this set, as if indexed by the full range.

        Implementations should avoid the copy that `get(np.arange(len(self)))`
        would make (fancy indexing always copies) - e.g. by returning the
        backing array directly when it already holds exactly this data.
        """
        raise NotImplementedError

    def full(self) -> DomainSubSet:
        """Return a DomainSubSet covering every position of this set."""
        return DomainSubSet.full(len(self))
