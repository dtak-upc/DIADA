from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.dataset.Dataset import Dataset
from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup


class OrderedPG(PredicateGroup):
    """Predicate group for ordered (numeric) columns: how the two rows of a
    pair compare in `column`.

    preds 0 -> equal, 1 -> not equal, 2 -> left < right, 3 -> left > right
    (defaults to computing all four).
    """

    def __init__(self, dataset: Dataset, column: str):
        super().__init__(dataset, column)
        self._values: npt.NDArray[np.number] = dataset.df[column].to_numpy()

    def numPredicates(self) -> int:
        return 4

    def pgEval(
        self, domain: DomainSet, x: Optional[DomainSubSet] = None
    ) -> List[npt.NDArray[np.uint8]]:
        if x is None:
            # full domain: use getFull() so domain.get() doesn't have to
            # copy via fancy indexing on np.arange(n)
            pairs = domain.getFull()
        else:
            pairs = domain.get(x.indices())

        left = self._values[pairs[:, 0]]
        right = self._values[pairs[:, 1]]
        same = (left == right).astype(np.uint8)
        less = (left < right).astype(np.uint8)
        return [same, less]

    def predPositions(
        self, variables: List[npt.NDArray[np.uint8]], preds: Optional[set] = None
    ) -> Dict[int, npt.NDArray[np.intp]]:
        if preds is None:
            preds = {0, 1, 2, 3}
        same, less = variables
        same = same.astype(bool)
        less = less.astype(bool)
        greater = ~(less | same)  # not (< or =), i.e. left > right

        result: Dict[int, npt.NDArray[np.intp]] = {}
        if 0 in preds:
            result[0] = np.where(same)[0].astype(np.intp)
        if 1 in preds:
            result[1] = np.where(~same)[0].astype(np.intp)
        if 2 in preds:
            result[2] = np.where(less)[0].astype(np.intp)
        if 3 in preds:
            result[3] = np.where(greater)[0].astype(np.intp)
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
