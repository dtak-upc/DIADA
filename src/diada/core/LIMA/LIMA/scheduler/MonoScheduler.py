from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.scheduler.Scheduler import Scheduler
from diada.core.LIMA.LIMA.scheduler.Schedule import (
    Schedule,
    adjacentNodes,
    nodesAboveThreshold,
    buildTree,
    invertTree,
)


class MonoScheduler(Scheduler):
    """Forces single-step sampling: returns the same schedule LIMAScheduler
    would (built from `state.candidateEdges`) the first time `schedule` is
    called, then an empty Schedule on every call after that.
    """

    def __init__(self) -> None:
        self._used = False

    def schedule(self, state: LatticeState, threshold: float) -> Schedule:
        if self._used:
            return Schedule()
        self._used = True

        nodes = adjacentNodes(state.candidateEdges)
        nodes = nodesAboveThreshold(nodes, state, threshold)
        tree = buildTree(nodes, state)
        children = invertTree(tree, state)
        return Schedule(tree, children)
