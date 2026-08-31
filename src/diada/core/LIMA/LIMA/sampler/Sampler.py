from abc import ABC, abstractmethod
from typing import List

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.sampler.Sample import Sample
from diada.core.LIMA.LIMA.scheduler.Schedule import Schedule


class Sampler(ABC):
    """Abstract: draws n samples from a domain following a schedule."""

    @abstractmethod
    def sample(
        self,
        sched: Schedule,
        domain: Domain,
        predicate_groups: List[PredicateGroup],
        state: LatticeState,
        n: int,
    ) -> Sample:
        raise NotImplementedError
