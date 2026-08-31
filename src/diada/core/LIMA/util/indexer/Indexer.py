from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import IndexLike
from diada.core.LIMA.util.map.NDMap import NDMap
from diada.core.LIMA.util.map.NpNDMap import NpNDMap
from diada.core.LIMA.util.list.NDList import NDList
from diada.core.LIMA.util.list.NpNDList import NpNDList


class Indexer(ABC):
    """Abstract map from arbitrary ints to sequentially-assigned unique IDs.

    IDs are assigned in the order ints are first seen via `add`, starting
    at 0. IDs are never reused or removed (no removal support for now).

    `map_cls`/`list_cls` are the NDMap/NDList implementations an
    implementation builds its internal id-lookup/reverse-lookup storage
    from; they're stored here so every implementation shares the same
    constructor shape, defaulting to the numpy-backed implementations.
    """

    def __init__(self, map_cls: Type[NDMap] = NpNDMap, list_cls: Type[NDList] = NpNDList) -> None:
        self.map_cls: Type[NDMap] = map_cls
        self.list_cls: Type[NDList] = list_cls

    @abstractmethod
    def __len__(self) -> int:
        """Number of distinct ints registered so far."""
        raise NotImplementedError

    @abstractmethod
    def add(self, x: IndexLike) -> None:
        """Register `x` (a single int, or an array of ints), assigning a
        fresh ID to any value not already seen. Values that are already
        registered are left untouched.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, x: IndexLike) -> npt.NDArray[np.integer]:
        """Return the ID(s) assigned to `x`. `x` must already be registered."""
        raise NotImplementedError
