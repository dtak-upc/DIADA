from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple

from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.util.stats.betaDist import BetaDist


class LatticeState(ABC):
    """Abstract mutable belief state over the lattice.

    Exposes exactly what outside classes (Scheduler/Sampler/LIMA
    implementations) actually use: `lattice`, `soundEdges`, `candidateEdges`,
    and the methods below. Everything else (e.g. how sound/candidate edges
    get maintained) is an implementation detail of a concrete subclass.
    """

    lattice: Lattice
    soundEdges: Set[Lattice.Edge]
    candidateEdges: Set[Lattice.Edge]

    @abstractmethod
    def getDist(self, n: Lattice.Node) -> BetaDist:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, results: Dict[Lattice.Node, Tuple[int, int]], threshold: float
    ) -> None:
        """Add a sampling round's results (as produced by a Sampler - just
        its `counts` dict, to avoid importing the sampler classes here) to
        each node's distribution, then refresh sound/candidate edges to
        reflect the updated distributions.
        """
        raise NotImplementedError

    @abstractmethod
    def _minSoundZ(self, edge: Lattice.Edge) -> float:
        """The minimum (most negative) nodeDist4 z-score over every predicate
        pg in the edge's parent - how sound `edge` is at its strongest
        supporting comparison, on the same scale a soundness threshold
        compares against.
        """
        raise NotImplementedError
