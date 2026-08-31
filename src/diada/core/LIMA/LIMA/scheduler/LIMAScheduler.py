from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.scheduler.Scheduler import Scheduler
from diada.core.LIMA.LIMA.scheduler.Schedule import (
    Schedule,
    adjacentNodes,
    nodesAboveThreshold,
    buildTree,
    invertTree,
)


class LIMAScheduler(Scheduler):

    def schedule(self, state: LatticeState, threshold: float) -> Schedule:
        """Build this round's sampling tree from the nodes adjacent to
        `state.candidateEdges` whose nodeGrad exceeds `threshold`."""
        nodes = adjacentNodes(state.candidateEdges)
        nodes = nodesAboveThreshold(nodes, state, threshold)
        tree = buildTree(nodes, state)
        children = invertTree(tree, state)
        return Schedule(tree, children)
