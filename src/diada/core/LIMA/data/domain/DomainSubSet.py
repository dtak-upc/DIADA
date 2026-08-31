from typing import Optional

import numpy as np
import numpy.typing as npt

SPARSE_THRESHOLD = 0.1  # use index list when < 10% of positions match


class DomainSubSet:
    """Adaptive sparse/dense subset of positions in a DomainSet of size `n`.

    Only needs `n` (the domain set's length), not the domain set itself.
    Dense: bits packed into _packed (uint8, MSB-first). AND is a byte-level bitop.
    Sparse: sorted position indices in _indices (intp). AND is intersect1d.
    Exactly one of _packed / _indices is set; the other is None.
    """

    def __init__(
        self,
        n:       int,
        packed:  Optional[npt.NDArray[np.uint8]],
        indices: Optional[npt.NDArray[np.intp]],
        length:  int,
    ) -> None:
        self._n       = n
        self._packed  = packed   # set iff dense
        self._indices = indices  # set iff sparse
        self._len     = length

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def full(n: int) -> 'DomainSubSet':
        return DomainSubSet(n, np.packbits(np.ones(n, dtype=np.bool_)), None, n)

    @staticmethod
    def _make(n: int, indices: npt.NDArray[np.intp]) -> 'DomainSubSet':
        """Choose representation from a sorted index array."""
        k = len(indices)
        if k < n * SPARSE_THRESHOLD:
            return DomainSubSet(n, None, indices.astype(np.intp), k)
        mask = np.zeros(n, dtype=np.bool_)
        if k:
            mask[indices] = True
        return DomainSubSet(n, np.packbits(mask), None, k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def indices(self) -> npt.NDArray[np.intp]:
        """Sorted array of matching positions into the domain set."""
        if self._indices is not None:
            return self._indices
        if self._packed is not None:
            return np.where(np.unpackbits(self._packed)[:self._n])[0].astype(np.intp)
        assert False

    def __len__(self) -> int:
        return self._len

    def __and__(self, other: 'DomainSubSet') -> 'DomainSubSet':
        n: int = self._n

        # Dense & Dense — byte-level AND
        # np.bitwise_count counts set bits per byte; padding bits are 0 so sum is exact.
        if self._packed is not None and other._packed is not None:
            result: npt.NDArray[np.uint8] = self._packed & other._packed
            count: int = int(np.bitwise_count(result).sum())
            if count < n * SPARSE_THRESHOLD:
                idx: npt.NDArray[np.intp] = np.where(np.unpackbits(result)[:n])[0].astype(np.intp)
                return DomainSubSet(n, None, idx, count)
            return DomainSubSet(n, result, None, count)

        # Sparse & Sparse — sorted-array intersection
        if self._indices is not None and other._indices is not None:
            idx = np.intersect1d(self._indices, other._indices, assume_unique=True)
            return DomainSubSet._make(n, idx)

        # Mixed — check each sparse index against the packed bits of the dense side
        # bit i lives in byte i>>3, at position 7-(i&7) from LSB (MSB-first packing)
        sparse_idx: npt.NDArray[np.intp]
        packed:     npt.NDArray[np.uint8]
        if self._indices is not None:
            sparse_idx, packed = self._indices, other._packed  # type: ignore[assignment]
        else:
            sparse_idx, packed = other._indices, self._packed  # type: ignore[assignment]
        bits: npt.NDArray[np.uint8] = ((packed[sparse_idx >> 3] >> (7 - (sparse_idx & 7))) & np.uint8(1)).astype(np.uint8)
        new_idx: npt.NDArray[np.intp] = sparse_idx[bits.view(np.bool_)]
        return DomainSubSet._make(n, new_idx)
