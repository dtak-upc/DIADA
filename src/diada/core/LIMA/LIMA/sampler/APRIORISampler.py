from typing import Dict, List, Tuple

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.sampler.Sample import Sample, makeFullDomainSet, evaluatePredicateGroups
from diada.core.LIMA.LIMA.sampler.Sampler import Sampler
from diada.core.LIMA.LIMA.scheduler.Schedule import Schedule


class APRIORISampler(Sampler):
    """Apriori-style sampler: instead of following a precomputed schedule
    tree, it evaluates every predicate group directly (over the whole
    domain) and does its own bounded-depth DFS over the lattice, up to
    nodes with `k` active predicates.

    Since there's no schedule to say which combinations are worth visiting,
    it orders the search itself: every individual predicate's selectivity
    (fraction of the domain it *excludes* - low selectivity means it's
    frequently true) is computed up front, and predicates are visited most-
    frequently-true first. This mirrors the classic Apriori pruning
    intuition - combinations built from frequent predicates stay large
    enough to be worth exploring further, while rare predicates would only
    get rarer combined with others, so they're pushed later in the order.
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

        precomputed = evaluatePredicateGroups(
            dict(enumerate(predicate_groups)), domain_set, full
        )

        sorted_preds = self._sortPredicates(precomputed, n)

        result = Sample()
        root = state.lattice.getRoot()
        self._search(root, full, domain_set, sorted_preds, len(sorted_preds), precomputed, result)
        return result

    @staticmethod
    def _sortPredicates(
        precomputed: Dict[int, List[DomainSubSet]], n: int
    ) -> List[Tuple[int, int]]:
        """Every (pg, p) predicate, sorted by ascending selectivity (= 1 -
        match_rate), so frequently-true predicates come first."""
        selectivities = [
            (1.0 - len(subset) / n, pg, p)
            for pg, subsets in precomputed.items()
            for p, subset in enumerate(subsets)
        ]
        selectivities.sort(key=lambda t: t[0])
        return [(pg, p) for _, pg, p in selectivities]

    def _search(
        self,
        node: Lattice.Node,
        x: DomainSubSet,
        domain: DomainSet,
        sorted_preds: List[Tuple[int, int]],
        minIDX: int,
        precomputed: Dict[int, List[DomainSubSet]],
        result: Sample,
    ) -> None:
        a = len(x)
        b = len(domain) - a
        result.counts[node] = (a, b)

        if len(node.preds) >= self.k:
            return

        for idx in range(minIDX):
            pg, p = sorted_preds[idx]
            if pg in node.preds:
                continue
            child_x = x & precomputed[pg][p]
            child_node = node.to(pg, p).to()
            self._search(child_node, child_x, domain, sorted_preds, idx, precomputed, result)
