from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup

_WORD_BITS = 32


class EviSet:
    """Deduplicates domain elements by their predicate behaviour.

    Elements that evaluate identically across every predicate group's
    minimal variables (`pgEval`) are collapsed into one entry, with a count
    of how many elements share it - e.g. 1000 domain elements with the same
    behaviour become a single stored entry instead of 1000.
    """

    def __init__(
        self,
        predicate_groups: List[PredicateGroup],
        domain: DomainSet,
        x: DomainSubSet,
    ) -> None:
        self.predicate_groups = predicate_groups
        self.domain = domain
        self.x = x

        variables = self._evalVariables()
        self.numBits = len(variables)

        bits = self._stackBits(variables)
        self.packed = self._packBits(bits)

        self.patterns, self.counts = self._countUnique(self.packed)

        unpacked = self._unpackBits(self.patterns)
        self.behaviours: List[List[npt.NDArray[np.uint8]]] = self._splitByGroup(unpacked)

    def _evalVariables(self) -> List[npt.NDArray[np.uint8]]:
        """All pgEval variables across every predicate group, flattened into
        a single list (one entry per bit of behaviour). Also records how
        many bits each predicate group contributed, in `self._bitsPerGroup`,
        so the packed bits can later be split back up per group.
        """
        variables: List[npt.NDArray[np.uint8]] = []
        self._bitsPerGroup: List[int] = []
        for pg in self.predicate_groups:
            pg_variables = pg.pgEval(self.domain, self.x)
            variables.extend(pg_variables)
            self._bitsPerGroup.append(len(pg_variables))
        return variables

    @staticmethod
    def _stackBits(variables: List[npt.NDArray[np.uint8]]) -> npt.NDArray[np.uint8]:
        """Stack the per-bit variable arrays into a single (n, m) 0/1 array."""
        return np.stack(variables, axis=1)

    @staticmethod
    def _packBits(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint32]:
        """Pack an (n, m) 0/1 array into (n, ceil(m / 32)) uint32 words."""
        n, m = bits.shape
        num_words = -(-m // _WORD_BITS)  # ceil division
        padded_width = num_words * _WORD_BITS

        padded = np.zeros((n, padded_width), dtype=np.uint32)
        padded[:, :m] = bits

        grouped = padded.reshape(n, num_words, _WORD_BITS)
        weights = 1 << np.arange(_WORD_BITS, dtype=np.uint32)
        return (grouped * weights).sum(axis=2, dtype=np.uint32)

    @staticmethod
    def _countUnique(
        packed: npt.NDArray[np.uint32],
    ) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.intp]]:
        """Distinct predicate-behaviour patterns and how many domain elements
        share each one."""
        return np.unique(packed, axis=0, return_counts=True)

    def _unpackBits(self, packed: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint8]:
        """Inverse of `_packBits`: (k, num_words) uint32 words back into a
        (k, numBits) 0/1 array."""
        k = packed.shape[0]
        weights = 1 << np.arange(_WORD_BITS, dtype=np.uint32)
        bits = ((packed[:, :, None] & weights) != 0).astype(np.uint8)
        bits = bits.reshape(k, -1)
        return bits[:, : self.numBits]

    def _splitByGroup(
        self, bits: npt.NDArray[np.uint8]
    ) -> List[List[npt.NDArray[np.uint8]]]:
        """Split a (k, numBits) bit array's columns back up per predicate
        group, using `self._bitsPerGroup`: for each group, a list with one
        (k,) array per variable it contributed, giving that variable's value
        for each of the k unique behaviours."""
        result: List[List[npt.NDArray[np.uint8]]] = []
        col = 0
        for count in self._bitsPerGroup:
            result.append([bits[:, col + i] for i in range(count)])
            col += count
        return result

    def fullIndex(self) -> npt.NDArray[np.intp]:
        """The index list covering every unique behaviour."""
        return np.arange(self.patterns.shape[0], dtype=np.intp)

    def totalCount(self, index: npt.NDArray[np.intp]) -> int:
        """Total number of domain elements represented by the unique
        behaviours in `index`."""
        return int(self.counts[index].sum())

    def filterByPredicate(
        self, index: npt.NDArray[np.intp], pg_idx: int, pred_idx: int
    ) -> npt.NDArray[np.intp]:
        """Restrict `index` (a list of unique-behaviour indices) to those
        satisfying predicate `pred_idx` of predicate group `pg_idx`.

        Gets that predicate group's variables restricted to `index`, uses
        its `predPositions` to find which of those satisfy the predicate,
        and maps the resulting local positions back through `index`.
        """
        variables = [var[index] for var in self.behaviours[pg_idx]]
        preds_positions = self.predicate_groups[pg_idx].predPositions(variables, {pred_idx})
        local_idx = preds_positions[pred_idx]
        return index[local_idx]
