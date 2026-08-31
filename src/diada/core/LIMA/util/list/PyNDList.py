from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike
from diada.core.LIMA.util.list.NDList import NDList, T


class PyNDList(NDList[T]):
    """Simplest possible NDList: backed by a plain python list."""

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        dtype: Optional[npt.DTypeLike] = np.int64,
        capacity: int = 8,
    ):
        super().__init__(shape, dtype, capacity)
        self._data: List[npt.NDArray[T]] = []

    def __len__(self) -> int:
        return len(self._data)

    @property
    def capacity(self) -> int:
        return len(self._data)

    def add(self, x: ArrayLike[T]) -> None:
        self._data.append(np.asarray(x, dtype=self.dtype).reshape(self.shape))

    def append(self, values: ArrayLike[T]) -> None:
        arr = np.asarray(values, dtype=self.dtype).reshape((-1,) + self.shape)
        for row in arr:
            self._data.append(row)

    def get(self, index: IndexLike) -> npt.NDArray[T]:
        if np.ndim(index) == 0:
            return self._data[int(index)]
        return np.stack([self._data[int(i)] for i in np.asarray(index)])

    def to_array(self) -> npt.NDArray[T]:
        if not self._data:
            return np.empty((0,) + self.shape, dtype=self.dtype)
        return np.stack(self._data)

    def clear(self) -> None:
        self._data.clear()
