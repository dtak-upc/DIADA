from typing import Dict, List, Optional, Tuple

from diada.core.LIMA.data.domain.Domain import Domain
from diada.core.LIMA.data.domain.DomainSet import DomainSet
from diada.core.LIMA.data.domain.DomainSubSet import DomainSubSet
from diada.core.LIMA.data.predicate.PredicateGroup import PredicateGroup
from diada.core.LIMA.LIMA.lattice.Lattice import Lattice
from diada.core.LIMA.LIMA.lattice.LatticeState import LatticeState
from diada.core.LIMA.LIMA.scheduler.Schedule import Schedule


class Sample:
    """Observed (successes, failures) counts per lattice node, from one sampling round."""

    def __init__(self, counts: Optional[Dict[Lattice.Node, Tuple[int, int]]] = None) -> None:
        self.counts: Dict[Lattice.Node, Tuple[int, int]] = counts if counts is not None else {}


def makeFullDomainSet(domain: Domain, n: int) -> Tuple[DomainSet, DomainSubSet]:
    """Instantiate a size-n DomainSet from `domain`, along with the
    DomainSubSet covering all of it."""
    domain_set = domain.makeDomainSet(n)
    return domain_set, domain_set.full()


def estimateReach(sched: Schedule, state: LatticeState, n: int, npgs: int) -> List[float]:
    """Estimate, per predicate group, the total number of domain elements
    expected to reach nodes in the schedule tree through that group.

    Walks `sched`'s tree from the root (reach = n); for each child edge
    (pg, p) of a node, the reach estimate for that child is
    `reach_at_node * state.getDist(node).mean`, which is added to
    `counts[pg]` and propagated as the reach for that child in turn.
    """
    counts = [0.0] * npgs
    root = state.lattice.getRoot()
    stack: List[Tuple[Lattice.Node, float]] = [(root, float(n))]
    while stack:
        node, reach = stack.pop()
        mean = state.getDist(node).mean
        for pg, p in sched.children.get(node, ()):
            estimate = reach * mean
            counts[pg] += estimate
            child = node.to(pg, p).to()
            stack.append((child, estimate))
    return counts


def partitionPredicateGroups(
    counts: List[float], n: int, factor: float = 0.01
) -> Tuple[List[int], List[int]]:
    """Decide which predicate groups to precompute (evaluate over the whole
    domain once, memoize, and AND with the DomainSubSet at each node) versus
    filter on demand.

    `counts[pg]` is treated as an estimate of how many times pg will need to
    be evaluated; if that exceeds `n * factor`, it's worth precomputing.

    Returns (precomputed, on_demand): lists of predicate group indices.
    """
    threshold = n * factor
    precomputed = [pg for pg, count in enumerate(counts) if count > threshold]
    on_demand = [pg for pg, count in enumerate(counts) if count <= threshold]
    return precomputed, on_demand


def evaluatePredicateGroups(
    pgs: Dict[int, PredicateGroup], domain: DomainSet, x: DomainSubSet
) -> Dict[int, List[DomainSubSet]]:
    """Evaluate every predicate of every predicate group in `pgs` over `x`.

    `pgs` is keyed by predicate group id. Returns a dict with the same keys,
    each mapping to a list with one DomainSubSet per predicate of that group
    (2 for UnorderedPG, 4 for OrderedPG - `eval`'s default `preds`), ordered
    by predicate id - i.e. exactly the format `filterDomain`'s `precomputed`
    argument expects.
    """
    result: Dict[int, List[DomainSubSet]] = {}
    for pg_id, pg in pgs.items():
        evaluated = pg.eval(domain, x)
        result[pg_id] = [evaluated[p] for p in sorted(evaluated)]
    return result


def evaluatePredicate(
    pg: PredicateGroup, p: int, domain: DomainSet, x: DomainSubSet
) -> DomainSubSet:
    """Evaluate a single predicate `p` of `pg` over `x`.

    Used when exploring the lattice tree and the added predicate wasn't
    among the precomputed ones.
    """
    return pg.eval(domain, x, {p})[p]


def filterDomain(
    x: DomainSubSet,
    pg_id: int,
    pg: PredicateGroup,
    p: int,
    domain: DomainSet,
    precomputed: Dict[int, List[DomainSubSet]],
) -> DomainSubSet:
    """Filter `x` by predicate `p` of predicate group `pg` (with id `pg_id`).

    If `pg_id` was precomputed (present in `precomputed`, e.g. via
    `evaluatePredicateGroups`), just AND `x` with the precomputed subset for
    `p` instead of evaluating the predicate again.
    """
    if pg_id in precomputed:
        return x & precomputed[pg_id][p]
    return evaluatePredicate(pg, p, domain, x)


def _exploreDFS(
    node: Lattice.Node,
    x: DomainSubSet,
    domain: DomainSet,
    sched: Schedule,
    predicate_groups: List[PredicateGroup],
    precomputed: Dict[int, List[DomainSubSet]],
    result: Sample,
) -> None:
    """DFS the schedule tree from `node`/`x`, recording (a, b) = (len(x),
    len(domain) - len(x)) - the new BetaDist observation for each node -
    into `result`, then recursing into each child after filtering `x` by
    that child edge's (pg, p).
    """
    a = len(x)
    b = len(domain) - a
    result.counts[node] = (a, b)

    for pg, p in sched.children.get(node, ()):
        child_node = node.to(pg, p).to()
        child_x = filterDomain(x, pg, predicate_groups[pg], p, domain, precomputed)
        _exploreDFS(child_node, child_x, domain, sched, predicate_groups, precomputed, result)


def explore(
    domain: DomainSet,
    full: DomainSubSet,
    sched: Schedule,
    state: LatticeState,
    predicate_groups: List[PredicateGroup],
    precomputed: Dict[int, List[DomainSubSet]],
) -> Sample:
    """Runner: DFS `sched`'s tree starting at the lattice root with the full
    DomainSubSet `full`, returning the resulting Sample.
    """
    result = Sample()
    root = state.lattice.getRoot()
    _exploreDFS(root, full, domain, sched, predicate_groups, precomputed, result)
    return result
