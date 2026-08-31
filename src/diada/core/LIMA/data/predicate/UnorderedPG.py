from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.dataset.Dataset import Dataset
from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup


class UnorderedPG(PredicateGroup):
    """Predicate group for categoric (unordered) columns: whether both rows
    of a pair have the same value in `column`.

    preds 0 -> pairs with the same value, preds 1 -> pairs with different
    values (defaults to computing both).
    """

    def __init__(self, dataset: Dataset, column: str):
        super().__init__(dataset, column)
        # `.to_numpy()` on a category-dtype Series re-materializes the
        # actual values from their codes on every call (never a view, since
        # Categorical isn't backed by a plain ndarray of values) - cache the
        # integer codes once instead. Codes are already a plain numpy array,
        # so this is cheap, and comparing small ints in eval() is faster
        # than comparing the underlying values (often strings) directly.
        self._codes: npt.NDArray[np.integer] = dataset.df[column].cat.codes.to_numpy()

    def numPredicates(self) -> int:
        return 2

    def pgEval(
        self, domain: DomainSet, x: Optional[DomainSubSet] = None
    ) -> List[npt.NDArray[np.uint8]]:
        if x is None:
            # full domain: use getFull() so domain.get() doesn't have to
            # copy via fancy indexing on np.arange(n)
            pairs = domain.getFull()
        else:
            pairs = domain.get(x.indices())

        left = self._codes[pairs[:, 0]]
        right = self._codes[pairs[:, 1]]
        same = (left == right).astype(np.uint8)  # raw equality, doesn't care about missing values
        return [same]

    def predPositions(
        self, variables: List[npt.NDArray[np.uint8]], preds: Optional[set] = None
    ) -> Dict[int, npt.NDArray[np.intp]]:
        if preds is None:
            preds = {0, 1}
        (same,) = variables
        same = same.astype(bool)

        result: Dict[int, npt.NDArray[np.intp]] = {}
        if 0 in preds:
            result[0] = np.where(same)[0].astype(np.intp)
        if 1 in preds:
            result[1] = np.where(~same)[0].astype(np.intp)
        return result

    def eval(
        self,
        domain: DomainSet,
        x: Optional[DomainSubSet] = None,
        preds: Optional[set] = None,
    ) -> Dict[int, DomainSubSet]:
        n = len(domain)
        positions = np.arange(n, dtype=np.intp) if x is None else x.indices()

        variables = self.pgEval(domain, x)
        preds_positions = self.predPositions(variables, preds)
        return self.makeSubsets(n, positions, preds_positions)
