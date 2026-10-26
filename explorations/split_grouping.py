"""
Grouping that cuts a triangle A-B-C at B into {A, B} and {B, C} when B explains the relation
between A and C, instead of DIADA's connected components (dataset_builder.build_mn_datasets),
which always keep A-B-C whole.

The rule:
  - a triangle A-B, B-C, A-C whose A-C is explained by B (the mediation test pruned A-C) is
    split: A and C may not share a group, B goes with both
  - a triangle whose A-C is not explained by any B stays whole
  - columns connected without a triangle (an open path A-B-C, a longer chain, ...) stay
    together, as in DIADA's connected components
So two columns may share a group if they are in the same connected component and the edge
between them was not pruned as explained by a third column. The groups are the maximal sets
in which every two columns may be together (maximal cliques of that "compatible" graph). A
column can be in several groups (the hub B is in {A, B} and {B, C}); with nothing pruned the
groups are exactly the connected components.

For MN, the target's dataset is the union of the groups containing the target; every other
group is its own dataset; columns with no edge go to "leftover", as in build_mn_datasets.

Nothing under src/diada/ is modified.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Set

import networkx as nx
import pandas as pd

from diada.core.dataset_builder import _ensure_target, filter_columns


def split_groups(graph: nx.Graph, separated: Set[frozenset]) -> List[List[str]]:
    """Groups of `graph`'s columns. `separated`: pairs A-C whose triangle edge was pruned as
    explained by a third column B."""
    compatible = nx.Graph()
    for component in nx.connected_components(graph):
        compatible.add_nodes_from(component)
        compatible.add_edges_from((x, y) for x, y in itertools.combinations(sorted(component), 2)
                                  if frozenset((x, y)) not in separated)
    return sorted((sorted(c) for c in nx.find_cliques(compatible)), key=lambda g: (-len(g), g))


def target_group(groups: List[List[str]], target: str) -> List[str]:
    """Columns of the target's MN dataset: the union of the groups containing the target."""
    return sorted(set().union(*(g for g in groups if target in g))) if any(target in g for g in groups) else [target]


def build_mn_split_datasets(groups: List[List[str]], base_dataset: pd.DataFrame, target_column: str) -> Dict[str, pd.DataFrame]:
    """The split-grouping counterpart of dataset_builder.build_mn_datasets:
    {"target_cluster": df, "cluster_2": df, ..., "leftover": df}, where target_cluster is the
    union of the groups containing the target and cluster_i are the other groups (largest
    first). Groups may overlap, so a column can be in several datasets."""
    datasets: Dict[str, pd.DataFrame] = {}
    cols = filter_columns(target_group(groups, target_column), base_dataset)
    datasets["target_cluster"] = _ensure_target(base_dataset[cols].copy(), base_dataset, target_column)
    for i, g in enumerate((g for g in groups if target_column not in g), start=2):
        cols = filter_columns(g, base_dataset)
        datasets[f"cluster_{i}"] = _ensure_target(base_dataset[cols].copy(), base_dataset, target_column)
    in_groups = set().union(*groups) if groups else set()
    leftover = [c for c in base_dataset.columns if c not in in_groups]
    datasets["leftover"] = _ensure_target(base_dataset[leftover].copy(), base_dataset, target_column)
    return datasets
