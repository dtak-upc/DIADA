from typing import Type

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import IndexLike
from diada.core.LIMA.util.indexer.Indexer import Indexer
from diada.core.LIMA.util.map.NDMap import NDMap
from diada.core.LIMA.util.map.NpNDMap import NpNDMap
from diada.core.LIMA.util.list.NDList import NDList
from diada.core.LIMA.util.list.NpNDList import NpNDList


class NPIndexer(Indexer):
    """Basic Indexer: an NDMap (int -> id) plus an NDList (id -> int).

    Which concrete NDMap/NDList implementations back it is up to the
    caller - the fields are typed against the abstractions so any pair of
    implementations can be swapped in later. Defaults to the numpy-backed
    implementations.
    """

    def __init__(self, map_cls: Type[NDMap] = NpNDMap, list_cls: Type[NDList] = NpNDList):
        super().__init__(map_cls, list_cls)
        self._id_of: NDMap = self.map_cls(shape=(), value_dtype=np.int64)
        self._int_of: NDList = self.list_cls(shape=(), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._int_of)

    def add(self, x: IndexLike) -> None:
        keys = np.atleast_1d(np.asarray(x, dtype=np.int64))

        unseen = np.array([k not in self._id_of for k in keys])
        new_keys = keys[unseen]
        if new_keys.size == 0:
            return
        new_keys = np.unique(new_keys)  # de-duplicate within this batch

        start = len(self._int_of)
        new_ids = np.arange(start, start + new_keys.size)

        self._id_of.set(new_keys, new_ids)
        self._int_of.append(new_keys)

    def get(self, x: IndexLike) -> npt.NDArray[np.integer]:
        return self._id_of.get(x)
