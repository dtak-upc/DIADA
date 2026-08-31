"""
ARDA JoinGraph — Transitive Join Discovery

Parses a connections.csv file that encodes a schema graph, then performs a BFS traversal from the base table to produce 
an ordered, deduplicated list of CandidateJoin descriptors ready for the ARDA pipeline.

connections.csv format (tab- or comma-separated):
    fk_table    fk_column   pk_table    pk_column
    t0.csv      Key_0       t1.csv      Key_0
    t1.csv      Key_1       t2.csv      Key_1
    ...

The graph is treated as undirected: an edge (fk_table, fk_col) ↔ (pk_table, pk_col) means the two tables can 
be joined on those two columns regardless of which direction the FK/PK relationship runs.

BFS guarantees:
  - The base table is the root; its direct neighbours are joined first.
  - A table reachable via multiple paths is joined exactly once (the first time it is discovered).
  - Each join's base_key refers to a column that is present in the ACCUMULATED table at the time the join is executed, accounting for
    suffix renaming performed by join_tables().

Column-name tracking
--------------------
join_tables() adds a suffix (default "_aug") to a column from the foreign table only when that column name already exists 
in the accumulated table AND the suffixed name does not already exist.  We replicate this logic here so
that downstream join descriptors reference the correct column name.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(name: str, table_dir: str) -> pd.DataFrame:
    """Load table into a pandas dataframe (supports CSV and Parquet, searches recursively)."""

    def _find_file(base_dir: str, filename: str) -> str:
        """Search for a file recursively inside base_dir."""
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        raise FileNotFoundError(f"Table file not found in {base_dir}: {filename}")

    path = _find_file(table_dir, name) # Find file (search in table_dir and subdirectories)

    _, ext = os.path.splitext(path)

    if ext.lower() == ".csv":
        return pd.read_csv(path, sep=_detect_sep(path))
    elif ext.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _detect_sep(path: str) -> str:
    with open(path, "r") as fh:
        sample = fh.read(2048)
    return "\t" if sample.count("\t") > sample.count(",") else ","


def _simulate_suffix(foreign_cols: List[str], foreign_key: str, base_key: str, accumulated_cols: Set[str], 
                     suffix: str = "_aug") -> Dict[str, str]:
    """
    Mirror the rename logic in joins.join_tables():

        rename_map = {
            c: c + suffix
            for c in foreign.columns
            if c != foreign_key
            and (c + suffix) not in base.columns
            and c in base.columns
        }

    Returns a mapping  original_col_in_foreign -> actual_col_in_accumulated
    for every non-dropped column.  The foreign_key column is dropped when it
    differs from base_key (it would be a duplicate join key).
    """
    mapping: Dict[str, str] = {}
    for c in foreign_cols:
        if c == foreign_key:
            if foreign_key == base_key:
                # Same name on both sides → already in accumulated as base_key
                pass
            else:
                # Different names → foreign_key gets dropped after the join
                pass
            continue
        if c in accumulated_cols and (c + suffix) not in accumulated_cols:
            mapping[c] = c + suffix
        else:
            mapping[c] = c
    return mapping


# ---------------------------------------------------------------------------
# JoinGraph
# ---------------------------------------------------------------------------

class JoinGraph:
    """
    Build a join graph from a connections.csv file and traverse it via BFS
    starting from the base table.

    Parameters
    ----------
    connections_path : path to connections CSV/TSV
    table_dir        : directory that contains the table CSV files
    suffix           : column suffix used when resolving name clashes
                       (must match the suffix passed to ARDA / join_tables)
    """

    def __init__( self, connections_path: str, table_dir: str = "."):
        self.connections_path = connections_path
        self.table_dir = table_dir

        self._table_cache: Dict[str, pd.DataFrame] = {}

        # Adjacency list:
        #   _adj[table_name] = list of (neighbour_table, my_col, neighbour_col)
        # where my_col is the join column in this table and neighbour_col is the join column in the neighbour.
        self._adj: Dict[str, List[Tuple[str, str, str]]] = {}
        self._connections: pd.DataFrame = pd.DataFrame()

        self._parse_connections()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candidate_joins(self, base_table_name: str) -> List[dict]:
        """
        BFS from `base_table_name`.  Returns an ordered list of
        CandidateJoin dicts ready to be passed to ARDA.fit_transform().

        Each dict contains:
            table       : pd.DataFrame  (loaded from disk)
            base_key    : str           (column in the *accumulated* table)
            foreign_key : str           (column in `table`)
            _table_name : str           (filename, for diagnostics)
            _depth      : int           (BFS depth, for diagnostics)
        """
        return self._bfs(base_table_name)

    def load_base_table(self, base_table_name: str) -> pd.DataFrame:
        """Load and return the base table DataFrame."""
        return self._load(base_table_name)

    def summary(self, base_table_name: str) -> str:
        """Pretty-print the BFS join order (no tables loaded)."""
        joins = self.get_candidate_joins(base_table_name)
        lines = [
            f"Join graph — BFS from '{base_table_name}'",
            f"  {len(joins)} table(s) reachable",
            "",
        ]
        for i, j in enumerate(joins, 1):
            lines.append(
                f"  {i:2d}. [{j['_depth']}] {j['_table_name']}"
                f"  via  accumulated[{j['base_key']}] ↔ foreign[{j['foreign_key']}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_connections(self) -> None:
        connections = pd.read_csv(self.connections_path, sep=_detect_sep(self.connections_path))
        connections.columns = [c.strip() for c in connections.columns] # Normalise column names: strip whitespace, lowercase for matching

        required = {"fk_table", "fk_column", "pk_table", "pk_column"}
        missing = required - set(connections.columns)
        if missing: # Check that the header is the appropiate one
            raise ValueError(
                f"connections file is missing columns: {missing}\n"
                f"Found: {list(connections.columns)}"
            )

        self._connections = connections

        # Build undirected adjacency
        for _, row in connections.iterrows():
            fk_t = str(row["fk_table"]).strip()
            fk_c = str(row["fk_column"]).strip()
            pk_t = str(row["pk_table"]).strip()
            pk_c = str(row["pk_column"]).strip()

            self._adj.setdefault(fk_t, []).append((pk_t, fk_c, pk_c))
            self._adj.setdefault(pk_t, []).append((fk_t, pk_c, fk_c))

    # ------------------------------------------------------------------
    # BFS traversal
    # ------------------------------------------------------------------

    def _bfs(self, base_table_name: str) -> List[dict]:
        """
        BFS from base_table_name.

        State tracked per iteration:
          visited       : set of table names already joined
          accumulated   : set of column names in the accumulated table so far
          col_origin    : maps (table_name, original_col) -> actual col name
                          in accumulated table (accounting for suffixes)
        """
        visited: Set[str] = {base_table_name}
        result: List[dict] = []

        # Initialise accumulated column set from the base table
        base_df = self._load(base_table_name)
        accumulated_cols: Set[str] = set(base_df.columns)

        # col_origin[(table, col)] = name of that column in accumulated table
        col_origin: Dict[Tuple[str, str], str] = {(base_table_name, c): c for c in base_df.columns}

        # BFS queue items: (table_name, depth)
        queue: deque[Tuple[str, int]] = deque([(base_table_name, 0)])

        while queue:
            current_table, depth = queue.popleft()

            for (neighbour, my_col, neighbour_col) in self._adj.get(current_table, []):
                if neighbour in visited:
                    continue
                visited.add(neighbour)

                # Resolve the base_key: what is my_col called in the accumulated table right now?
                accumulated_base_key = col_origin.get((current_table, my_col), my_col)

                # If the column was dropped during a previous join (it was the foreign_key and differed from base_key),
                # it won't be in accumulated_cols. In that case, try to find it under the base_key name of that join step.
                if accumulated_base_key not in accumulated_cols:
                    # Fall back: search for any column that was mapped from (current_table, my_col) including indirect paths
                    fallback = self._find_col_in_accumulated(my_col, current_table, col_origin, accumulated_cols)
                    if fallback is None: # Column genuinely unavailable — skip this edge
                        visited.discard(neighbour)
                        continue
                    accumulated_base_key = fallback

                neighbour_df = self._load(neighbour) # Load the neighbour table

                # Simulate how join_tables() will rename neighbour columns
                rename = _simulate_suffix(
                    foreign_cols=list(neighbour_df.columns),
                    foreign_key=neighbour_col,
                    base_key=accumulated_base_key,
                    accumulated_cols=accumulated_cols,
                    suffix=neighbour,
                )

                # Update accumulated_cols and col_origin
                for orig_c, actual_c in rename.items():
                    accumulated_cols.add(actual_c)
                    col_origin[(neighbour, orig_c)] = actual_c

                # The base_key column was already in accumulated; record it for the neighbour as well (both sides of the join share it)
                if neighbour_col != accumulated_base_key:
                    # foreign_key is dropped; record that it maps to base_key
                    col_origin[(neighbour, neighbour_col)] = accumulated_base_key
                else:
                    col_origin[(neighbour, neighbour_col)] = accumulated_base_key

                result.append(
                    {
                        "table": neighbour_df,
                        "base_key": accumulated_base_key,
                        "foreign_key": neighbour_col,
                        "suffix": neighbour,
                        "_table_name": neighbour,
                        "_from_table": current_table,
                        "_depth": depth + 1,
                    }
                )

                queue.append((neighbour, depth + 1))

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self, name: str) -> pd.DataFrame:
        if name not in self._table_cache:
            self._table_cache[name] = _load_table(name, self.table_dir)
        return self._table_cache[name]

    @staticmethod
    def _find_col_in_accumulated(col: str, table: str, col_origin: Dict[Tuple[str, str], str], accumulated_cols: Set[str]) -> Optional[str]:
        """
        Try variations of `col` that might exist in accumulated_cols: exact match, with suffix, etc.
        """
        if col in accumulated_cols: # Direct name still present?
            return col
        # Registered under (table, col)?
        registered = col_origin.get((table, col))
        if registered and registered in accumulated_cols:
            return registered
        return None
