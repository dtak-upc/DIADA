from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike
from diada.core.LIMA.util.list.NDList import NDList, T


class NpNDList(NDList[T]):
    """NDList backed by a single numpy array, doubling capacity as needed.

    Storage is shaped (capacity,) + shape, with the element count as the
    outer dimension, so elements stay contiguous.
    """

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        dtype: Optional[npt.DTypeLike] = np.int64,
        capacity: int = 8,
    ):
        super().__init__(shape, dtype, capacity)
        self._size = 0
        self._data: npt.NDArray[T] = np.empty((capacity,) + self.shape, dtype=dtype)

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._data.shape[0]

    def _grow(self, min_capacity: int) -> None:
        capacity = self._data.shape[0] or 1
        while capacity < min_capacity:
            capacity *= 2
        grown = np.empty((capacity,) + self.shape, dtype=self.dtype)
        grown[: self._size] = self._data[: self._size]
        self._data = grown

    def add(self, x: ArrayLike[T]) -> None:
        if self._size == self.capacity:
            self._grow(self._size + 1)
        self._data[self._size] = x
        self._size += 1

    def append(self, values: ArrayLike[T]) -> None:
        arr = np.asarray(values, dtype=self.dtype)
        if arr.shape == self.shape:
            arr = arr[np.newaxis, ...]  # a single element promoted to a batch of one

        n = arr.shape[0]
        if self._size + n > self.capacity:
            self._grow(self._size + n)
        self._data[self._size:self._size + n] = arr
        self._size += n

    def get(self, index: IndexLike) -> npt.NDArray[T]:
        if np.ndim(index) == 0:
            return self._data[int(index)]
        return self._data[np.asarray(index)]

    def to_array(self) -> npt.NDArray[T]:
        return self._data[: self._size]

    def clear(self) -> None:
        self._size = 0
