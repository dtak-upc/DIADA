from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike

V = TypeVar("V", bound=np.generic)


class NDMap(ABC, Generic[V]):
    """Abstract map from integer keys to fixed-shape numpy-array values.

    `shape` is the shape of a single stored value (mirrors NDList). Keys are
    always integers - a single int/numpy integer for one entry, or an
    integer array to address several entries at once (e.g. in `get`) - same
    as plain numpy indexing.
    """

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        value_dtype: Optional[npt.DTypeLike] = np.int64,
        bits: int = 3,
    ):
        self.shape: tuple[int, ...] = tuple(shape)
        self.value_dtype: Optional[npt.DTypeLike] = value_dtype
        self._capacity: int = 1<<bits

    @abstractmethod
    def __len__(self) -> int:
        """Number of key/value pairs currently stored."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Number of key/value pairs the backing storage can currently hold."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: IndexLike, value: ArrayLike[V]) -> None:
        """Insert or update the value for `key`.

        `key` may be a plain int, a numpy integer, or an integer array (to
        set several entries at once); `value` may be a real array, a numpy
        scalar, or a plain int - same as numpy itself accepts either an int
        or an array to index/assign with.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: IndexLike) -> npt.NDArray[V]:
        """Return the value(s) stored for `key`.

        A single int (or numpy integer) returns one value, shaped
        `self.shape`; an array of keys returns a batch, shaped
        (n,) + self.shape - same as plain numpy indexing.
        """
        raise NotImplementedError

    @abstractmethod
    def remove(self, key: IndexLike) -> None:
        """Remove the entry (or entries) for `key`."""
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: IndexLike) -> bool:
        """Whether `key` currently has a stored value."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        raise NotImplementedError
