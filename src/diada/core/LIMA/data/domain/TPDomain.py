from typing import Optional

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.domain.TPDomainSet import TPDomainSet
from diada.core.LIMA.util.random.RandZ2 import RandZ2


class TPDomain(Domain[TPDomainSet]):
    """Domain of tuple pairs drawn from a fixed range (e.g. a dataset's row count)."""

    def __init__(self, range_: int, rz2: Optional[RandZ2] = None):
        self.range_ = range_
        self.rz2 = rz2 if rz2 is not None else RandZ2()

    def makeDomainSet(self, size: int) -> TPDomainSet:
        pairs = self.rz2.uniformGrid(size, self.range_)
        return TPDomainSet(pairs)
