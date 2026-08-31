from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.dataset.Dataset import Dataset
from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet


class PredicateGroup(ABC):
    """Abstract group of predicates, defined over a single dataset column,
    evaluated over pairs of dataset rows.
    """

    def __init__(self, dataset: Dataset, column: str):
        self.dataset = dataset
        self.column = column

    @abstractmethod
    def numPredicates(self) -> int:
        """Number of predicates this group evaluates (the size of `eval`'s
        default `preds`)."""
        raise NotImplementedError

    @abstractmethod
    def pgEval(
        self, domain: DomainSet, x: Optional[DomainSubSet] = None
    ) -> List[npt.NDArray[np.uint8]]:
        """Compute this group's minimal set of underlying boolean variables
        over the pairs at positions `x` of `domain` (e.g. just `same` for an
        unordered group, `same`/`less` for an ordered one), aligned with
        `x`'s positions (or the full domain if `x` is None).

        `eval()` derives its full predicate set from these variables.
        """
        raise NotImplementedError

    @abstractmethod
    def predPositions(
        self, variables: List[npt.NDArray[np.uint8]], preds: Optional[set] = None
    ) -> Dict[int, npt.NDArray[np.intp]]:
        """From this group's `pgEval` variables, the local index arrays
        (positions into those variables, not into any DomainSet) of the
        elements satisfying each requested predicate.

        This is the part of `eval()` that doesn't need a DomainSet/DomainSubSet
        at all - just the variable arrays - so it can be reused directly on
        e.g. an EviSet's deduplicated per-behaviour variables.
        """
        raise NotImplementedError

    @staticmethod
    def makeSubsets(
        n: int,
        positions: npt.NDArray[np.intp],
        preds_positions: Dict[int, npt.NDArray[np.intp]],
    ) -> Dict[int, DomainSubSet]:
        """Build the final per-predicate DomainSubSets from `predPositions`'s
        local index arrays, mapped through the domain-level `positions` this
        evaluation covered.
        """
        return {
            p: DomainSubSet._make(n, positions[local_idx])
            for p, local_idx in preds_positions.items()
        }

    @abstractmethod
    def eval(
        self, domain: DomainSet, x: Optional[DomainSubSet]=None, preds: Optional[set]=None
    ) -> Dict[int, DomainSubSet]:
        """Evaluate the predicate group over the pairs at positions `x` of
        `domain`, restricted to the predicates named in `preds`.

        `x=None` means the full domain - implementations should use
        `domain.getFull()` for that case rather than materializing a full
        index array and indexing with it.

        Returns a dict mapping each requested predicate (from `preds`) to
        the DomainSubSet satisfying it. Implementations should run `pgEval`,
        then `predPositions`, then `makeSubsets`, in that order.
        """
        raise NotImplementedError
