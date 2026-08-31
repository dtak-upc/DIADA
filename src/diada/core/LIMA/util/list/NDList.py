from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike

T = TypeVar("T", bound=np.generic)


class NDList(ABC, Generic[T]):
    """Abstract growable list of fixed-shape numpy elements.

    `shape` is the shape of a single element; the backing storage is
    conceptually shaped (capacity,) + shape, with the element count as the
    outer dimension.
    """

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        dtype: Optional[npt.DTypeLike] = np.int64,
        capacity: int = 8,
    ):
        self.shape: tuple[int, ...] = tuple(shape)
        self.dtype: Optional[npt.DTypeLike] = dtype
        self._capacity: int = capacity

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements currently stored."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Number of elements the backing storage can currently hold."""
        raise NotImplementedError

    @abstractmethod
    def add(self, x: ArrayLike[T]) -> None:
        """Append a single element (of shape `self.shape`) to the end.

        `x` may be a real array, a numpy scalar, or a plain int - same as
        numpy itself accepts either an int or an array to index/assign with.
        """
        raise NotImplementedError

    @abstractmethod
    def append(self, values: ArrayLike[T]) -> None:
        """Append a batch of elements, shaped (n,) + self.shape."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index: IndexLike) -> npt.NDArray[T]:
        """Return the element(s) at `index`.

        A single int (or numpy integer) returns one element, shaped
        `self.shape`; an array of indices returns a batch, shaped
        (n,) + self.shape - same as plain numpy indexing.
        """
        raise NotImplementedError

    @abstractmethod
    def to_array(self) -> npt.NDArray[T]:
        """Return a view of all stored elements, shaped (len(self),) + self.shape."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all elements."""
        raise NotImplementedError
