from typing import List, Optional, Set, Union

import numpy as np
import numpy.typing as npt

from diada.core.LIMA.data.dataset.Dataset import Dataset
from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.domain.TPDomain import TPDomain
from diada.core.LIMA.data.predicate.OrderedPG import OrderedPG
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.data.predicate.UnorderedPG import UnorderedPG
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.lattice.LIMALatticeState import LIMALatticeState
from diada.core.LIMA.LIMA.sampler.LIMASampler import LIMASampler
from diada.core.LIMA.LIMA.sampler.Sampler import Sampler
from diada.core.LIMA.LIMA.scheduler.LIMAScheduler import LIMAScheduler
from diada.core.LIMA.LIMA.scheduler.Scheduler import Scheduler

_PRED_OPS = ["=", "!=", "<", ">"]  # predicate id -> operator symbol, shared by Unordered/OrderedPG


class LIMA:
    """Main controller: owns the dataset, its per-attribute predicate groups,
    and the lattice belief state over them."""

    def __init__(
        self,
        dataset: Union[str, Dataset],
        scheduler: Optional[Scheduler] = None,
        sampler: Optional[Sampler] = None,
        approx: float = 1e-6,
        devs: float = 4,
    ):
        # accepts either a CSV path (built into a Dataset here) or an
        # already-constructed Dataset (e.g. from Dataset.expand())
        self.dataset: Dataset = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        self.predicate_groups: List[PredicateGroup] = []
        self.buildPreds()

        self.domain: Domain = TPDomain(len(self.dataset.df))
        self.lattice: Lattice = Lattice([pg.numPredicates() for pg in self.predicate_groups])
        self.state: LatticeState = LIMALatticeState(self.lattice)

        self.scheduler: Scheduler = scheduler if scheduler is not None else LIMAScheduler()
        self.sampler: Sampler = sampler if sampler is not None else LIMASampler()

        # sole owners of these defaults - Scheduler.schedule/LatticeState.update
        # take them as required parameters, not defaults of their own
        self.approx = approx  # scheduler threshold for filtering nodes (nodeGrad cutoff)
        self.devs = devs  # soundness test threshold, in z-score standard deviations

    def buildPreds(self) -> None:
        """Build one PredicateGroup per column: an
        UnorderedPG for categoric columns, an OrderedPG for everything else.
        """
        self.predicate_groups = [
                UnorderedPG(self.dataset, column)
                if dtype == "category"
                else OrderedPG(self.dataset, column)

            for column, dtype in self.dataset.dtypes.items()
        ]

    def getStepSize(self, iteration: int) -> int:
        """Sample size for `iteration`: doubles from 2**10 up to a cap of
        2**17, reached once iteration >= 7."""
        return 1 << (10 + min(iteration, 7))

    def step(self, iteration: int) -> bool:
        """Run one round: schedule, sample, update.

        Returns True immediately if the schedule comes back empty (nothing
        left to explore), else runs the sample/update and returns False.
        """
        sched = self.scheduler.schedule(self.state, self.approx)
        if not sched.parents:
            return True

        n = self.getStepSize(iteration)
        result = self.sampler.sample(sched, self.domain, self.predicate_groups, self.state, n)
        self.state.update(result.counts, self.devs)
        return False

    def deadEnds(self) -> Set[Lattice.Node]:
        """Sound edges whose to-node has never seen a positive observation
        (BetaDist.a == 1, i.e. still exactly the uninformative prior's a)."""
        return {
            edge.to()
            for edge in self.state.soundEdges
            if self.state.getDist(edge.to()).a == 1
        }

    def nodeText(self, n: Lattice.Node) -> str:
        """Render node n as e.g. '!(colA=colA & colB!=colB)': the negated
        conjunction of its predicates, each written as column<op>column."""
        return "!(" + " & ".join(
            f"{self.dataset.columns[pg]}{_PRED_OPS[p]}{self.dataset.columns[pg]}"
            for pg, p in n.preds.items()
        ) + ")"

    def deadEndsText(self) -> List[str]:
        """Text rendering of every dead-end node from `deadEnds()`."""
        return [self.nodeText(n) for n in self.deadEnds()]

    def relatedness(self) -> npt.NDArray[np.float64]:
        """Relatedness matrix between predicate groups (attributes).

        For every sound edge parent->child adding predicate group c to
        parent (parent's active groups being e.g. {a, b, ...}), this uses
        `state._minSoundZ(edge)` and, for every predicate group x already
        active in parent, keeps the largest -z seen for entry [x, c].

        A larger entry[x, y] means a stronger (more negative z) soundness
        signal was found for some edge that adds y on top of a node already
        containing x.
        """
        npgs = len(self.predicate_groups)
        matrix: npt.NDArray[np.float64] = np.full((npgs, npgs), 0.0)

        for edge in self.state.soundEdges:
            c = edge.pg
            value = -self.state._minSoundZ(edge)
            parent = edge.fr()
            for x in parent.preds:
                if value > matrix[x, c]:
                    matrix[x, c] = value

        return matrix
