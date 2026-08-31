from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.util.stats.betaDist import nodeGrad


class Schedule:
    """A sample schedule: the sampling tree for this round.

    `parents` maps each node to be sampled to its parent's pg (-1 for a
    root); `children` is its inverse - each node to the set of (pg, p)
    edges leading to its children in the tree.
    """

    def __init__(
        self,
        parents: Optional[Dict[Lattice.Node, int]] = None,
        children: Optional[Dict[Lattice.Node, Set[Tuple[int, int]]]] = None,
    ) -> None:
        self.parents: Dict[Lattice.Node, int] = parents if parents is not None else {}
        self.children: Dict[Lattice.Node, Set[Tuple[int, int]]] = (
            children if children is not None else {}
        )


def bestParentPg(state: LatticeState, candidates: List[Tuple[int, Lattice.Node]]) -> int:
    """Given `candidates` (pg, node.fr(pg).fr()) pairs, the pg whose parent
    has the smallest mean.
    """
    return min(candidates, key=lambda c: state.getDist(c[1]).mean)[0]


def _ensurePath(
    node: Lattice.Node,
    state: LatticeState,
    roots: Set[Lattice.Node],
    tree: Dict[Lattice.Node, int],
) -> None:
    """Make sure `node` has a path to the root recorded in `tree`.

    `roots` must include the true lattice root (or another node already
    guaranteed to terminate the recursion) - `node.preds` being empty is not
    checked separately here.
    """
    if node in tree:
        return
    if node in roots:
        tree[node] = -1
        return

    candidates = [(pg, node.fr(pg).fr()) for pg in node.preds]
    for _, parent in candidates:
        _ensurePath(parent, state, roots, tree)

    tree[node] = bestParentPg(state, candidates)


def buildTree(nodes: Set[Lattice.Node], state: LatticeState) -> Dict[Lattice.Node, int]:
    """Build a tree (node -> parent pg) containing a path from every node in
    `nodes` to the root.
    """
    tree: Dict[Lattice.Node, int] = {}
    roots = {state.lattice.getRoot()}
    for node in nodes:
        _ensurePath(node, state, roots, tree)
    return tree


def invertTree(
    tree: Dict[Lattice.Node, int], state: LatticeState
) -> Dict[Lattice.Node, Set[Tuple[int, int]]]:
    """Invert a parent tree (node -> parent pg) into a children map: each
    node to the set of (pg, p) edges leading to its children.
    """
    children: Dict[Lattice.Node, Set[Tuple[int, int]]] = defaultdict(set)
    for node, pg in tree.items():
        if pg == -1:
            continue
        edge = node.fr(pg)  # the edge parent -> node
        parent = edge.fr()
        children[parent].add((edge.pg, edge.p))
    return children


def adjacentNodes(edges: Set[Lattice.Edge]) -> Set[Lattice.Node]:
    """All nodes touched by `edges`: both the `fr()` and `to()` endpoint of each."""
    nodes: Set[Lattice.Node] = set()
    for e in edges:
        nodes.add(e.fr())
        nodes.add(e.to())
    return nodes


def nodesAboveThreshold(
    nodes: Set[Lattice.Node], state: LatticeState, threshold: float
) -> Set[Lattice.Node]:
    """Nodes whose distribution's nodeGrad exceeds `threshold`."""
    return {n for n in nodes if nodeGrad(state.getDist(n)) > threshold}
