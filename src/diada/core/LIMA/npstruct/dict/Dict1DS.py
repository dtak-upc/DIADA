import numpy as np

_EMPTY = np.iinfo(np.int64).min  # slot is unused

# Knuth's multiplicative hashing constant: 2^32 / golden ratio, rounded to
# the nearest odd number. Odd so it's invertible mod 2^32 (no information is
# lost), which is what makes the multiplication mix bits well.
_HASH_MULTIPLIER = np.uint32(2654435769)


class Dict1DS:
    """Simple integer->integer hash map: no probing.

    Each key maps to exactly one slot. If that slot is taken by a different
    key, we just grow (double capacity, rehash everything) and retry -
    trusting the hash to spread keys out rather than chaining/probing to a
    neighboring slot.

    The slot is chosen from the *high* bits of `key * _HASH_MULTIPLIER`
    (computed in 32-bit space), not the raw low bits of the key. Using the
    key's own low bits directly is what let 10**9, 2**40 etc. collide before
    (they're all multiples of 256, so their low bits are all zero); the
    multiply spreads every input bit's influence across the whole 32-bit
    product, and the high bits of a multiplication are the ones that are
    fully mixed (the low bits of a product are dominated by the low bits of
    the inputs), so reading the index off the top avoids that same failure
    mode, including for small keys, which no longer just hash to a run of
    leading zeros.
    """

    def __init__(self, k: int = 4):
        assert k <= 32  # index is derived from a 32-bit hash, can't exceed that
        self._k = k
        self._size = 0
        capacity = 1 << k
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty(capacity, dtype=np.int64)

    def __len__(self) -> int:
        return self._size

    def _capacity(self) -> int:
        return self._keys.shape[0]

    def _index(self, key: int) -> int:
        # plain python ints: arbitrary precision, so both the 32-bit
        # truncation and the multiply wrap correctly with no overflow
        # warning (numpy's fixed-width uint32 would warn on the multiply,
        # even though the wraparound is exactly what we want here)
        truncated = key & 0xFFFFFFFF
        h = (truncated * int(_HASH_MULTIPLIER)) & 0xFFFFFFFF
        return h >> (32 - self._k)

    def _indexN(self, keys: np.ndarray) -> np.ndarray:
        truncated = (keys & 0xFFFFFFFF).astype(np.uint32)
        with np.errstate(over="ignore"):  # wraparound mod 2^32 is intentional
            h = (truncated * _HASH_MULTIPLIER) & np.uint32(0xFFFFFFFF)
        return (h >> np.uint32(32 - self._k)).astype(np.int64)

    def add(self, key: int, value: int):
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

    def addN(self, keys: np.ndarray, values: np.ndarray):
        """Vectorized batch insert/update.

        Same no-probing rule as `add`: if inserting would put two different
        keys in the same slot (whether one is already in the table, or two
        new keys in this batch target the same slot), we grow and retry the
        whole batch rather than searching for a neighboring slot.
        """
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        if keys.shape[0] == 0:
            return

        # last-value-wins for duplicate keys within the batch, same as
        # calling add() for each element in order would do
        dedup = dict(zip(keys.tolist(), values.tolist()))
        keys = np.array(list(dedup.keys()), dtype=np.int64)
        values = np.array(list(dedup.values()), dtype=np.int64)

        while True:
            idx = self._indexN(keys)
            slot_keys = self._keys[idx]

            match_mask = slot_keys == keys
            empty_mask = (~match_mask) & (slot_keys == _EMPTY)
            conflict_mask = ~(match_mask | empty_mask)  # slot taken by a different key

            insert_idx = idx[empty_mask]
            has_duplicate_targets = insert_idx.size != np.unique(insert_idx).size

            if np.any(conflict_mask) or has_duplicate_targets:
                # a real hash collision, either with an existing entry or
                # between two distinct keys in this batch: grow and retry
                self._grow()
                continue

            if np.any(match_mask):
                self._values[idx[match_mask]] = values[match_mask]

            if insert_idx.size:
                self._keys[insert_idx] = keys[empty_mask]
                self._values[insert_idx] = values[empty_mask]
                self._size += insert_idx.size

            return

    def remove(self, key: int):
        idx = self._index(key)
        if self._keys[idx] == key:
            self._keys[idx] = _EMPTY
            self._size -= 1
            return
        raise KeyError(key)

    def removeN(self, keys: np.ndarray):
        # duplicates in the request map to the same single slot; collapse
        # them so we don't try to clear (and decrement) it twice
        keys = np.unique(np.asarray(keys, dtype=np.int64))
        if keys.shape[0] == 0:
            return

        idx = self._indexN(keys)
        match_mask = self._keys[idx] == keys

        if not np.all(match_mask):
            raise KeyError(keys[~match_mask].tolist())

        self._keys[idx] = _EMPTY
        self._size -= idx.size

    def get(self, key: int) -> int:
        idx = self._index(key)
        if self._keys[idx] == key:
            return int(self._values[idx])
        raise KeyError(key)

    def _grow(self):
        old_keys = self._keys
        old_values = self._values

        self._k += 1
        capacity = 1 << self._k
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty(capacity, dtype=np.int64)
        self._size = 0

        live = old_keys != _EMPTY
        for key, value in zip(old_keys[live], old_values[live]):
            self.add(int(key), int(value))
