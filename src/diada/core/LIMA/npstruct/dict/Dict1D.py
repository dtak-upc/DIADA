import numpy as np

_EMPTY = np.iinfo(np.int64).min       # slot was never used
_DELETED = np.iinfo(np.int64).min + 1  # slot held an entry that was removed (tombstone)


class Dict1D:

    def __init__(self, k: int = 4, load_factor: float = 0.5):
        self._k = k
        self._load_factor = load_factor
        self._size = 0    # number of live entries
        self._filled = 0  # live entries + tombstones
        capacity = 1 << k
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty(capacity, dtype=np.int64)

    def __len__(self) -> int:
        return self._size

    def _capacity(self) -> int:
        return self._keys.shape[0]

    def _index(self, key: int) -> int:
        return key & (self._capacity() - 1)

    def _indexN(self, keys: np.ndarray) -> np.ndarray:
        return keys & (self._capacity() - 1)

    def add(self, key: int, value: int):
        capacity = self._capacity()
        idx = self._index(key)
        first_tombstone = -1
        while True:
            slot_key = self._keys[idx]
            if slot_key == key:
                self._values[idx] = value
                return
            if slot_key == _EMPTY:
                insert_at = idx if first_tombstone == -1 else first_tombstone
                if self._keys[insert_at] == _EMPTY:
                    self._filled += 1
                self._keys[insert_at] = key
                self._values[insert_at] = value
                self._size += 1
                break
            if slot_key == _DELETED and first_tombstone == -1:
                first_tombstone = idx
            idx = (idx + 1) % capacity

        if self._filled / self._capacity() > self._load_factor:
            self._grow()

    def addN(self, keys: np.ndarray, values: np.ndarray):
        """Vectorized batch insert/update.

        Duplicate keys within the same call are fine: the last occurrence in
        `keys` determines the stored value, matching plain dict semantics.
        """
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        n = keys.shape[0]
        if n == 0:
            return

        # make sure capacity can absorb the whole batch without growing mid-probe
        while (self._filled + n) / self._capacity() > self._load_factor:
            self._grow()

        capacity = self._capacity()
        idx = self._indexN(keys)
        first_tombstone = np.full(n, -1, dtype=np.int64)
        pending = np.arange(n)

        while pending.size > 0:
            cur_idx = idx[pending]
            slot_keys = self._keys[cur_idx]
            cur_keys = keys[pending]

            match_mask = slot_keys == cur_keys
            empty_mask = (~match_mask) & (slot_keys == _EMPTY)
            deleted_mask = (~match_mask) & (~empty_mask) & (slot_keys == _DELETED)
            other_mask = ~(match_mask | empty_mask | deleted_mask)

            # key already present -> just overwrite its value
            if np.any(match_mask):
                self._values[cur_idx[match_mask]] = values[pending[match_mask]]

            # chain terminates in an empty slot -> insert here (reusing an
            # earlier tombstone for this element if one was seen). Several
            # distinct pending keys can target the very same slot in the same
            # round (they share a probe path); only one may claim it here -
            # the rest ("losers") stay pending and re-examine the same slot
            # next round, where it will no longer look empty to them.
            loser_pos = pending[:0]
            if np.any(empty_mask):
                emp_pos = pending[empty_mask]
                emp_idx = cur_idx[empty_mask]
                emp_tomb = first_tombstone[emp_pos]
                insert_idx = np.where(emp_tomb != -1, emp_tomb, emp_idx)

                _, first_occurrence = np.unique(insert_idx, return_index=True)
                win_mask = np.zeros(insert_idx.shape[0], dtype=bool)
                win_mask[first_occurrence] = True

                win_pos = emp_pos[win_mask]
                win_idx = insert_idx[win_mask]
                newly_filled = int(np.count_nonzero(self._keys[win_idx] == _EMPTY))
                self._filled += newly_filled
                self._keys[win_idx] = keys[win_pos]
                self._values[win_idx] = values[win_pos]
                self._size += win_pos.size

                loser_pos = emp_pos[~win_mask]

            # tombstone along the chain -> remember the first one, keep probing
            if np.any(deleted_mask):
                del_pos = pending[deleted_mask]
                del_idx = cur_idx[deleted_mask]
                need_record = first_tombstone[del_pos] == -1
                first_tombstone[del_pos[need_record]] = del_idx[need_record]

            # elements blocked by a tombstone or a different live key must keep
            # probing forward; slot-collision losers retry the same slot as-is
            advance_pos = pending[deleted_mask | other_mask]
            idx[advance_pos] = (idx[advance_pos] + 1) % capacity
            pending = np.concatenate([advance_pos, loser_pos])

        if self._filled / self._capacity() > self._load_factor:
            self._grow()

    def remove(self, key: int):
        capacity = self._capacity()
        idx = self._index(key)
        while True:
            slot_key = self._keys[idx]
            if slot_key == key:
                self._keys[idx] = _DELETED
                self._size -= 1
                return
            if slot_key == _EMPTY:
                raise KeyError(key)
            idx = (idx + 1) % capacity

    def removeN(self, keys: np.ndarray):
        keys = np.asarray(keys, dtype=np.int64)
        n = keys.shape[0]
        if n == 0:
            return

        capacity = self._capacity()
        idx = self._indexN(keys)
        pending = np.arange(n)
        missing = []

        while pending.size > 0:
            cur_idx = idx[pending]
            slot_keys = self._keys[cur_idx]
            cur_keys = keys[pending]

            match_mask = slot_keys == cur_keys
            empty_mask = (~match_mask) & (slot_keys == _EMPTY)
            other_mask = ~(match_mask | empty_mask)

            if np.any(match_mask):
                matched_idx = cur_idx[match_mask]
                self._keys[matched_idx] = _DELETED
                self._size -= int(np.count_nonzero(match_mask))

            if np.any(empty_mask):
                # chain terminates without a match -> these keys aren't present
                missing.extend(cur_keys[empty_mask].tolist())

            keep_pos = pending[other_mask]
            idx[keep_pos] = (idx[keep_pos] + 1) % capacity
            pending = keep_pos

        if missing:
            raise KeyError(missing)

    def get(self, key: int) -> int:
        capacity = self._capacity()
        idx = self._index(key)
        while True:
            slot_key = self._keys[idx]
            if slot_key == key:
                return int(self._values[idx])
            if slot_key == _EMPTY:
                raise KeyError(key)
            idx = (idx + 1) % capacity

    def _grow(self):
        old_keys = self._keys
        old_values = self._values

        self._k += 1
        capacity = 1 << self._k
        self._keys = np.full(capacity, _EMPTY, dtype=np.int64)
        self._values = np.empty(capacity, dtype=np.int64)
        self._size = 0
        self._filled = 0

        live = (old_keys != _EMPTY) & (old_keys != _DELETED)
        for key, value in zip(old_keys[live], old_values[live]):
            self.add(int(key), int(value))
