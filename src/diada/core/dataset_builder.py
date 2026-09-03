"""
From a DIADA soundness table, build the cleaned dataset(s) for UN / MN
cleaning (univariate-noise / multivariate-noise -- renamed from the old
"CF1"/"CF2" naming, which was confusing carried-over jargon from an earlier
version; the goal each strategy serves is named directly instead: UN drops
columns that don't correlate with anything else, MN additionally splits out
sets of noisy, intra-correlated columns as their own partition(s)).

Was benchmarks_generation/clean_benchmarks.py, generalized so nothing DIADA
discards is thrown away silently:

  - UN used to keep only the columns related (above threshold) to *something*
    and drop the rest. `build_un_datasets` now also returns that discarded
    complement as a "leftover" dataset.
  - MN used to keep only the connected component containing the target and
    drop every other cluster. `build_mn_datasets` now returns one dataset
    per component (the target's own, plus every other cluster) and a
    "leftover" dataset of columns that had no relation above threshold to
    anything at all (never became a graph node).

Every returned dataset gets `target_column` appended (from the original
`base_dataset`) if it isn't already present, so each one can be fed into ARDA
independently -- including, deliberately, the "off-topic" clusters and
leftovers: e.g. does ARDA correctly score a spurious cluster's own dataset
low against the real target, rather than being fooled by its internal
correlation structure?

Exception: the `include_synth_targets=True` branch of `build_un_datasets`
mirrors the original script's behavior of deliberately dropping the *real*
target_column from the cleaned output (that variant's actual prediction
target is one of the synthetic_target_* columns instead) -- target-appending
is intentionally skipped there so that design isn't silently undone.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

DEFAULT_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Soundness graph (unchanged from build_graph_from_soundness)
# ---------------------------------------------------------------------------

def build_soundness_graph(soundness_df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> nx.Graph:
    """
    Build an undirected graph from a soundness triple DataFrame.

    Edges with score <= threshold or self-loops are excluded.
    When multiple rows describe the same pair, only the highest score is kept.
    """
    expected_cols = {"column_1", "column_2", "soundness"}
    if not expected_cols.issubset(soundness_df.columns):
        raise ValueError(f"DataFrame must contain columns {expected_cols}")

    df = soundness_df.copy()
    df["column_1"] = df["column_1"].astype(str)
    df["column_2"] = df["column_2"].astype(str)
    df["soundness"] = pd.to_numeric(df["soundness"], errors="coerce")

    df = df.dropna(subset=["soundness"])
    df = df[df["column_1"] != df["column_2"]]  # drop self-loops
    df = df[df["soundness"] > threshold]  # apply threshold

    # Keep only the highest score per unordered pair
    df["_pair"] = df.apply(lambda r: frozenset((r["column_1"], r["column_2"])), axis=1)
    df = df.sort_values("soundness", ascending=False).drop_duplicates("_pair")

    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_edge(row["column_1"], row["column_2"], weight=float(row["soundness"]))

    return graph


def filter_columns(selected_columns: List[str], df: pd.DataFrame, include_synth_targets: bool = False) -> List[str]:
    """Return only those column names that actually exist in df."""
    if not include_synth_targets:
        return [col for col in selected_columns if col in df.columns.to_list()]
    else:
        return [col for col in selected_columns if col in df.columns.to_list() or col in ["synthetic_target_1", "synthetic_target_2"]]


def _ensure_target(df: pd.DataFrame, base_dataset: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Append `target_column` from `base_dataset` if `df` doesn't already have it, so every
    generated dataset can be run through ARDA on its own."""
    if target_column in df.columns or target_column not in base_dataset.columns:
        return df
    df = df.copy()
    df[target_column] = base_dataset[target_column].reindex(df.index)
    return df


# ---------------------------------------------------------------------------
# UN (univariate noise): single "kept" set (+ its complement)
# ---------------------------------------------------------------------------

def build_un_datasets(soundness_df: pd.DataFrame, base_dataset: pd.DataFrame, target_column: str,
                       threshold: float = DEFAULT_THRESHOLD, include_synth_targets: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Returns {"kept": df, "leftover": df}.

    `kept` = every column with a soundness relation above `threshold` to
    anything (i.e. every node of the soundness graph) -- this is exactly
    `generate_benchmark_clean_cf1`'s old selection (old naming; see module
    docstring for the CF1/CF2 -> UN/MN rename).
    `leftover` = every base_dataset column NOT selected into `kept`.
    """
    graph = build_soundness_graph(soundness_df, threshold)
    selected_columns = sorted(graph.nodes())
    kept_cols = filter_columns(selected_columns, base_dataset, include_synth_targets)

    kept = base_dataset[kept_cols].copy()
    leftover = base_dataset[[c for c in base_dataset.columns if c not in kept_cols]].copy()

    if include_synth_targets:
        # This variant's real prediction target is one of the synthetic_target_*
        # columns, not target_column -- match the original script and keep
        # target_column out of the cleaned output. Left untouched (no
        # target-appending) since that design is deliberate here.
        if target_column in kept.columns:
            kept = kept.drop(columns=[target_column])
    else:
        kept = _ensure_target(kept, base_dataset, target_column)
        leftover = _ensure_target(leftover, base_dataset, target_column)

    _log_un_selection(kept, target_column)
    return {"kept": kept, "leftover": leftover}


def _log_un_selection(kept: pd.DataFrame, target_column: str) -> None:
    cols = kept.columns.to_list()
    noise_features = len([c for c in cols if "noise" in c or "shuffle" in c or "spurious" in c])
    print("Num of selected columns:", len(cols))
    print("Num of noisy columns selected:", noise_features)
    print("Was the target selected?", target_column in cols)


# ---------------------------------------------------------------------------
# MN (multivariate noise): one dataset per connected component (+ a leftover
# of isolated columns)
# ---------------------------------------------------------------------------

def build_mn_datasets(soundness_df: pd.DataFrame, base_dataset: pd.DataFrame, target_column: str,
                       threshold: float = DEFAULT_THRESHOLD) -> Dict[str, pd.DataFrame]:
    """
    Returns {"target_cluster": df, "cluster_2": df, ..., "leftover": df}.

    One dataset per connected component of the soundness graph: the
    component containing `target_column` is "target_cluster" (matching the
    old script's only output), the rest are "cluster_2", "cluster_3", ...
    ordered by size descending for determinism. "leftover" is every
    base_dataset column that never became a graph node at all (no relation
    above threshold to anything) -- the MN analogue of UN's leftover.
    """
    graph = build_soundness_graph(soundness_df, threshold)
    all_clusters = list(nx.connected_components(graph))

    target_cluster = next((c for c in all_clusters if target_column in c), {target_column})
    if target_column not in graph:
        print(f"Note: '{target_column}' has no relations >= {threshold}. Its cluster is just itself.")

    other_clusters = sorted(
        (c for c in all_clusters if c is not target_cluster and target_column not in c),
        key=len, reverse=True,
    )

    datasets: Dict[str, pd.DataFrame] = {}

    target_cols = filter_columns(list(target_cluster), base_dataset)
    datasets["target_cluster"] = _ensure_target(base_dataset[target_cols].copy(), base_dataset, target_column)
    print(f"Cluster for '{target_column}' contains {len(target_cols)} features.")

    for i, cluster in enumerate(other_clusters, start=2):
        cols = filter_columns(list(cluster), base_dataset)
        datasets[f"cluster_{i}"] = _ensure_target(base_dataset[cols].copy(), base_dataset, target_column)

    graph_cols = set().union(*all_clusters) if all_clusters else set()
    leftover_cols = [c for c in base_dataset.columns if c not in graph_cols]
    datasets["leftover"] = _ensure_target(base_dataset[leftover_cols].copy(), base_dataset, target_column)

    for name, dataset in datasets.items():
        noise_features = len([c for c in dataset.columns if "noise" in c or "shuffle" in c or "spurious" in c])
        print(f"[{name}] {len(dataset.columns)} columns, {noise_features} of them noisy")

    return datasets


# ---------------------------------------------------------------------------
# Optional visualization (opt-in interactive display; savefig is always safe)
# ---------------------------------------------------------------------------

def visualize_clusters(soundness_df: pd.DataFrame, target_column: str, output_path: str,
                        threshold: float = DEFAULT_THRESHOLD, show: bool = False) -> None:
    """
    Save a PNG of the soundness graph's connected components (target's
    cluster in red/orange, everything else in sky blue).

    `show=False` by default: the original script called `plt.show()`
    unconditionally, which blocks on an interactive matplotlib backend with
    no display (e.g. a fresh clone of this repo run headless/on a server).
    Pass `show=True` explicitly for interactive use (e.g. a notebook).
    """
    import matplotlib.pyplot as plt

    graph = build_soundness_graph(soundness_df, threshold)
    clusters = list(nx.connected_components(graph))
    target_cluster = next((c for c in clusters if target_column in c), set())

    node_colors = [
        "red" if node == target_column else
        "orange" if node in target_cluster else
        "skyblue"
        for node in graph.nodes()
    ]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, k=0.6, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=600, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.3)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    plt.title(f"Feature Clusters (Target: {target_column} in Red/Orange)")
    plt.axis("off")
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
