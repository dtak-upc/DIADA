from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.predicate.EviSet import EviSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.sampler.Sample import Sample, makeFullDomainSet
from diada.core.LIMA.LIMA.sampler.Sampler import Sampler
from diada.core.LIMA.LIMA.scheduler.Schedule import Schedule


class FPSampler(Sampler):
    """Frequent-pattern-style sampler: like APRIORISampler (its own bounded-
    depth DFS over the lattice, no reliance on a precomputed schedule tree),
    but instead of evaluating/filtering full DomainSubSets it first
    deduplicates domain elements by predicate behaviour via an EviSet, and
    explores/filters using that set's (much smaller) index lists of unique
    behaviours.

    Predicates are tried in their default order (no selectivity sorting),
    but the search is still tree-shaped via the same minIDX threading, so no
    node gets generated more than once.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def sample(
        self,
        sched: Schedule,
        domain: Domain,
        predicate_groups: List[PredicateGroup],
        state: LatticeState,
        n: int,
    ) -> Sample:
        domain_set, full = makeFullDomainSet(domain, n)
        eviset = EviSet(predicate_groups, domain_set, full)

        preds = self._defaultPreds(predicate_groups)

        result = Sample()
        root = state.lattice.getRoot()
        self._search(root, eviset.fullIndex(), n, eviset, preds, len(preds), result)
        return result

    @staticmethod
    def _defaultPreds(predicate_groups: List[PredicateGroup]) -> List[Tuple[int, int]]:
        """Every (pg, p) predicate in default order (no selectivity sorting)."""
        return [
            (pg_idx, p)
            for pg_idx, pg in enumerate(predicate_groups)
            for p in range(pg.numPredicates())
        ]

    def _search(
        self,
        node: Lattice.Node,
        index: npt.NDArray[np.intp],
        n: int,
        eviset: EviSet,
        preds: List[Tuple[int, int]],
        minIDX: int,
        result: Sample,
    ) -> None:
        a = eviset.totalCount(index)
        b = n - a
        result.counts[node] = (a, b)

        if len(node.preds) >= self.k:
            return

        for idx in range(minIDX):
            pg, p = preds[idx]
            if pg in node.preds:
                continue
            child_index = eviset.filterByPredicate(index, pg, p)
            child_node = node.to(pg, p).to()
            self._search(child_node, child_index, n, eviset, preds, idx, result)
