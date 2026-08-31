from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from diada.core.LIMA.data.domain.DomainSet import DomainSet

DS = TypeVar("DS", bound=DomainSet)


class Domain(ABC, Generic[DS]):
    """Abstract representation of an entire (uninstantiated) space.

    A Domain doesn't hold any data itself - it's just able to produce
    instantiated DomainSets (of a chosen size) that live in that space.
    Concrete subclasses fix `DS` to their own DomainSet implementation, so
    `makeDomainSet` returns that concrete type rather than the abstraction.
    """

    @abstractmethod
    def makeDomainSet(self, size: int) -> DS:
        """Instantiate a DomainSet of `size` elements from this domain."""
        raise NotImplementedError
