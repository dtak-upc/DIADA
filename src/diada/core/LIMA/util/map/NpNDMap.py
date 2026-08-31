from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.util.nptypes import ArrayLike, IndexLike
from diada.core.LIMA.util.map.NDMap import NDMap, V

_EMPTY = np.iinfo(np.int64).min  # slot is unused

# Knuth's multiplicative hashing constant: 2^32 / golden ratio, rounded to
# the nearest odd number. Odd so it's invertible mod 2^32 (no information is
# lost), which is what makes the multiplication mix bits well.
_HASH_MULTIPLIER = np.uint32(2654435769)


class NpNDMap(NDMap[V]):
    """NDMap backed by numpy arrays, no probing.

    Each key maps to exactly one slot, chosen from the *high* bits of
    `key * _HASH_MULTIPLIER` (computed in 32-bit space) rather than the
    key's own low bits - this spreads structured keys (e.g. multiples of a
    power of two) out instead of letting them collide directly. If a slot is
    already taken by a different key, we just grow (double capacity, rehash
    everything) and retry, trusting the hash to spread keys out rather than
    probing to a neighboring slot.
    """

    def __init__(
        self,
        shape: Sequence[int] = (1,),
        value_dtype: Optional[npt.DTypeLike] = np.int64,
        bits: int = 3,
    ):
        assert bits <= 32  # index is derived from a 32-bit hash, can't exceed that
        super().__init__(shape, value_dtype, bits)
        self._bits = bits
        self._size = 0
        capacity = 1 << bits
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values: npt.NDArray[V] = np.empty((capacity,) + self.shape, dtype=value_dtype)

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._keys.shape[0]

    def _index(self, key: int) -> int:
        # plain python ints: arbitrary precision, so both the 32-bit
        # truncation and the multiply wrap correctly with no overflow
        # warning (numpy's fixed-width uint32 would warn on the multiply,
        # even though the wraparound is exactly what we want here)
        truncated = key & 0xFFFFFFFF
        h = (truncated * int(_HASH_MULTIPLIER)) & 0xFFFFFFFF
        return h >> (32 - self._bits)

    def _set_one(self, key: int, value) -> None:
        while True:
            idx = self._index(key)
            slot_key = self._keys[idx]
            if slot_key == key:
                self._values[idx] = value
                return
            if slot_key == _EMPTY:
                self._keys[idx] = key
                self._values[idx] = value
                self._size += 1
                return
            # slot taken by a different key -> grow until it isn't
            self._grow()

    def set(self, key: IndexLike, value: ArrayLike[V]) -> None:
        if np.ndim(key) == 0:
            self._set_one(int(key), value)
            return
        keys = np.asarray(key, dtype=np.int64)
        values = np.asarray(value, dtype=self.value_dtype).reshape((-1,) + self.shape)
        for k, v in zip(keys, values):
            self._set_one(int(k), v)

    def get(self, key: IndexLike) -> npt.NDArray[V]:
        if np.ndim(key) == 0:
            idx = self._index(int(key))
            if self._keys[idx] == key:
                return self._values[idx]
            raise KeyError(int(key))
        return np.stack([self.get(int(k)) for k in np.asarray(key)])

    def remove(self, key: IndexLike) -> None:
        if np.ndim(key) == 0:
            idx = self._index(int(key))
            if self._keys[idx] == key:
                self._keys[idx] = _EMPTY
                self._size -= 1
                return
            raise KeyError(int(key))
        for k in np.asarray(key):
            self.remove(int(k))

    def __contains__(self, key: IndexLike) -> bool:
        if np.ndim(key) == 0:
            idx = self._index(int(key))
            return bool(self._keys[idx] == key)
        return all(self.__contains__(int(k)) for k in np.asarray(key))

    def clear(self) -> None:
        capacity = self.capacity
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty((capacity,) + self.shape, dtype=self.value_dtype)
        self._size = 0

    def _grow(self) -> None:
        old_keys = self._keys
        old_values = self._values

        self._bits += 1
        capacity = 1 << self._bits
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty((capacity,) + self.shape, dtype=self.value_dtype)
        self._size = 0

        live = old_keys != _EMPTY
        for key, value in zip(old_keys[live], old_values[live]):
            self._set_one(int(key), value)
