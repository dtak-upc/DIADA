from typing import Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike
from diada.core.LIMA.util.map.NDMap import NDMap, V


class PyNDMap(NDMap[V]):
    """Simplest possible NDMap: backed by a plain python dict."""

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        value_dtype: Optional[npt.DTypeLike] = np.int64,
        bits: int = 3,
    ):
        super().__init__(shape, value_dtype, bits)
        self._data: Dict[int, npt.NDArray[V]] = {}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def capacity(self) -> int:
        return len(self._data)

    def set(self, key: IndexLike, value: ArrayLike[V]) -> None:
        if np.ndim(key) == 0:
            self._data[int(key)] = np.asarray(value, dtype=self.value_dtype).reshape(self.shape)
            return
        values = np.asarray(value, dtype=self.value_dtype).reshape((-1,) + self.shape)
        for k, v in zip(np.asarray(key), values):
            self._data[int(k)] = v

    def get(self, key: IndexLike) -> npt.NDArray[V]:
        if np.ndim(key) == 0:
            return self._data[int(key)]
        return np.stack([self._data[int(k)] for k in np.asarray(key)])

    def remove(self, key: IndexLike) -> None:
        if np.ndim(key) == 0:
            del self._data[int(key)]
            return
        for k in np.asarray(key):
            del self._data[int(k)]

    def __contains__(self, key: IndexLike) -> bool:
        if np.ndim(key) == 0:
            return int(key) in self._data
        return all(int(k) in self._data for k in np.asarray(key))

    def clear(self) -> None:
        self._data.clear()
