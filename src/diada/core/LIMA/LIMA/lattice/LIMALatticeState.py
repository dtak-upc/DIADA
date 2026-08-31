import math
from collections import defaultdict
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.util.stats.betaDist import BetaDist, nodeDist4
from typing import Dict, Iterator, List, Set, Tuple


class LIMALatticeState(LatticeState):
    """Concrete LatticeState: stores a BetaDist per explored node and a
    cache of sound/candidate edges, maintained via a statistical soundness
    test as new sampling results come in.
    """

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice
        self._dists: Dict[Lattice.Node, BetaDist] = {}
        # Cached set of edges deemed sound by the statistical test.
        # Populated/updated externally
        self.soundEdges: Set[Lattice.Edge] = set()
        self.candidateEdges: Set[Lattice.Edge] = set()
        # Root always starts explored with uninformative prior
        self._dists[lattice.getRoot()] = BetaDist.empty()

        # Seed: every root -> depth-1 edge is sound, and every
        # depth-1 -> depth-2 edge is a candidate.
        root = lattice.getRoot()
        for pg in range(lattice.npgs):
            for p in range(lattice.pgs[pg]):
                edge = root.to(pg, p)
                self.soundEdges.add(edge)

                child = edge.to()
                for pg2 in range(lattice.npgs):
                    if pg2 == pg:
                        continue
                    for p2 in range(lattice.pgs[pg2]):
                        self.candidateEdges.add(child.to(pg2, p2))

    def getDist(self, n: Lattice.Node) -> BetaDist:
        return self._dists.get(n, BetaDist.empty())

    def isExplored(self, n: Lattice.Node) -> bool:
        return n in self._dists

    def explored(self) -> Iterator[Lattice.Node]:
        return iter(self._dists)

    def updateNode(self, n: Lattice.Node, a: int, b: int) -> None:
        """Add a new batch of (successes, failures) to a node's distribution."""
        if n not in self._dists:
            self._dists[n] = BetaDist.empty()
        self._dists[n].add(a, b)

    def update(
        self, results: Dict[Lattice.Node, Tuple[int, int]], threshold: float
    ) -> None:
        """Add a sampling round's results (as produced by a Sampler - just
        its `counts` dict, to avoid importing the sampler classes here) to
        each node's distribution, then refresh sound/candidate edges to
        reflect the updated distributions.
        """
        for n, (a, b) in results.items():
            self.updateNode(n, a, b)
        self._updateSoundEdges(threshold)

    def _updateSoundEdges(self, threshold: float) -> None:
        newly_sound: List[Lattice.Edge] = []
        for edge in list(self.candidateEdges):
            if self._isSound(edge, threshold):
                self.candidateEdges.discard(edge)
                self.soundEdges.add(edge)
                newly_sound.append(edge)
        for edge in newly_sound:
            self.propagateSound(edge)

    def getTreeParent(self, n: Lattice.Node) -> Lattice.Node:
        """Spanning-tree parent of n: remove the predicate whose removal gives the
        most selective (lowest meanLO) parent node."""
        best_pg = min(n.preds, key=lambda pg: self.getDist(n.fr(pg).fr()).meanLO)
        return n.fr(best_pg).fr()

    def initializeToDepth(self, k: int) -> None:
        """Pre-populate nodes and sound edges for all lattice nodes up to depth k.

        Every node at depth ≤ k gets an uninformative BetaDist(1,1), and every
        upward edge between adjacent layers is added to soundEdges. This lets
        the scheduler immediately propose depth-(k+1) candidates on the first step,
        treating the shallower layers as already validated.
        """

        visited: Set[Lattice.Node] = {self.lattice.getRoot()}
        frontier: List[Lattice.Node] = [self.lattice.getRoot()]
        for _ in range(k):
            next_frontier: List[Lattice.Node] = []
            for node in frontier:
                for pg in range(self.lattice.npgs):
                    if pg in node:
                        continue
                    for p in range(self.lattice.pgs[pg]):
                        edge = node.to(pg, p)
                        child = edge.to()
                        self.soundEdges.add(edge)
                        if child not in visited:
                            visited.add(child)
                            self._dists[child] = BetaDist.empty()
                            next_frontier.append(child)
            frontier = next_frontier

    def getTreeParentEdge(self, n: Lattice.Node) -> Lattice.Edge:
        """Upward edge from tree-parent to n (parent → n, adding best_pg)."""
        best_pg = min(n.preds, key=lambda pg: self.getDist(n.fr(pg).fr()).meanLO)
        parent = n.fr(best_pg).fr()
        return parent.to(best_pg, n.preds[best_pg])

    def _isSound(self, edge: Lattice.Edge, threshold: float) -> bool:
        """Edge parent→child is sound if, for every predicate pg in parent,
        nodeDist4(parent-pg, child-pg, parent, child) < -threshold.

        Each comparison asks: does this edge add more selectivity than the
        reference edge that lacks predicate pg?
        """
        parent = edge.fr()
        from1_dist = self.getDist(parent)
        to1 = edge.to()
        to1_dist = self.getDist(to1)
        for pg in parent.preds:
            subset = parent.fr(pg).fr()
            edge2 = subset.to(edge.pg, edge.p)
            to2 = edge2.to()
            from2_dist = self.getDist(subset)
            to2_dist = self.getDist(to2)
            z = nodeDist4(
                from1_dist,
                to1_dist,
                from2_dist,
                to2_dist,
            )
            if z >= -threshold:
                return False
            else:
                pass
        return True

    def _minSoundZ(self, edge: Lattice.Edge) -> float:
        """The minimum (most negative) nodeDist4 z-score over every predicate
        pg in parent - i.e. how sound `edge` is at its strongest supporting
        comparison, on the same scale `_isSound` thresholds against.
        """
        parent = edge.fr()
        from1_dist = self.getDist(parent)
        to1 = edge.to()
        to1_dist = self.getDist(to1)
        best = math.inf
        for pg in parent.preds:
            subset = parent.fr(pg).fr()
            edge2 = subset.to(edge.pg, edge.p)
            to2 = edge2.to()
            from2_dist = self.getDist(subset)
            to2_dist = self.getDist(to2)
            z = nodeDist4(
                from1_dist,
                to1_dist,
                from2_dist,
                to2_dist,
            )
            best = min(best, z)
        return best

    def _isCandidate(self, edge: Lattice.Edge) -> bool:
        """Edge parent→child is candidate if every parallel support edge is sound.

        For each predicate in parent, remove it and check that the same-predicate
        edge from that reduced parent is in soundEdges.
        """
        parent = edge.fr()
        for pg in parent.preds:
            if parent.fr(pg).fr().to(edge.pg, edge.p) not in self.soundEdges:
                return False
        return True

    def propagateSound(self, edge: Lattice.Edge) -> None:
        """Given newly-sound edge A→X, find supersets of A (excluding X) and register
        any new candidate edges that add the same predicate from those supersets."""
        parent = edge.fr()
        lattice = self.lattice
        for pg2 in range(lattice.npgs):
            if pg2 in parent.preds or pg2 == edge.pg:
                continue
            for p2 in range(lattice.pgs[pg2]):
                superset = parent.to(pg2, p2).to()
                candidate_edge = superset.to(edge.pg, edge.p)
                if self._isCandidate(candidate_edge):
                    self.candidateEdges.add(candidate_edge)
