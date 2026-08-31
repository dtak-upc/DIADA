from typing import List

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.sampler.Sampler import Sampler
from diada.core.LIMA.LIMA.sampler.Sample import (
    Sample,
    estimateReach,
    partitionPredicateGroups,
    makeFullDomainSet,
    evaluatePredicateGroups,
    explore,
)
from diada.core.LIMA.LIMA.scheduler.Schedule import Schedule


class LIMASampler(Sampler):

    def sample(
        self,
        sched: Schedule,
        domain: Domain,
        predicate_groups: List[PredicateGroup],
        state: LatticeState,
        n: int,
    ) -> Sample:
        """Draw `n` samples from `domain` as `sched` calls for, returning the observed counts per node.

        Estimates how often each predicate group will be evaluated, precomputes
        the ones that cross the worthwhile-to-precompute threshold, then DFS's
        the schedule tree to actually gather the (a, b) counts per node.
        """
        counts = estimateReach(sched, state, n, len(predicate_groups))
        precomputed_ids, _ = partitionPredicateGroups(counts, n)

        domain_set, full = makeFullDomainSet(domain, n)
        precomputed = evaluatePredicateGroups(
            {pg_id: predicate_groups[pg_id] for pg_id in precomputed_ids}, domain_set, full
        )

        return explore(domain_set, full, sched, state, predicate_groups, precomputed)
