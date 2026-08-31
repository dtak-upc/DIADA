import numpy as np
import numpy.typing as npt
from typing import Generic, TypeVar

T = TypeVar("T", bound=np.generic)


class List1D(Generic[T]):

    def __init__(self, dtype: npt.DTypeLike, capacity: int = 8):
        self._dtype = dtype
        self._size = 0
        self._data: npt.NDArray[T] = np.empty(capacity, dtype=dtype)

    def __len__(self) -> int:
        return self._size

    def _grow(self, min_capacity: int):
        capacity = self._data.shape[0]
        while capacity < min_capacity:
            capacity *= 2
        grown = np.empty(capacity, dtype=self._dtype)
        grown[:self._size] = self._data[:self._size]
        self._data = grown

    def add(self, value):
        if self._size == self._data.shape[0]:
            self._grow(self._size + 1)
        self._data[self._size] = value
        self._size += 1

    def append(self, values: npt.NDArray[T]):
        n = values.shape[0]
        if self._size + n > self._data.shape[0]:
            self._grow(self._size + n)
        self._data[self._size:self._size + n] = values
        self._size += n

    def get(self) -> npt.NDArray[T]:
        return self._data[:self._size]
