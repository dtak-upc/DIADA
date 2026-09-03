"""
ARDA – Automatic Relational Data Augmentation

This module offers two ways to run ARDA:

1. ``ARDA.fit_transform_full_table``  — PRIMARY, used throughout this project
   Takes a single table that already contains every candidate column (e.g.
   one of the pre-built noisy benchmarks in ``data/``, or a user's own CSV
   via ``diada.pipeline``) and runs only ARDA's feature-selection stage
   (RIFS, Sections 5-6 of the paper): rank the candidate columns, then
   train a model with the top-ranked features. There is no join
   discovery or execution here — "join" is assumed to already have
   happened upstream (or not to be needed at all).

   Two modes, via the ``num_features`` argument: pass an int to train
   exactly ONE model with that many top-ranked features (what the
   single-CSV "system" in ``pipeline.py`` / the ``diada-clean`` CLI does
   by default); omit it (``num_features=None``) to instead sweep every
   feature count from 1 up to ``max_features``, tracing out the
   accuracy/#features curve this project's paper results are built from
   (what ``arda/run_benchmarks.py`` and
   ``experiments/run_paper_experiments.py`` do).

2. ``ARDA.fit_transform`` / ``ARDA.fit_transform_from_connections`` — ORIGINAL ARDA
   The full pipeline described in the ARDA paper, end to end: coreset
   construction (Section 3.1) -> join-graph discovery/execution
   (Section 4) -> feature selection (Sections 5-6) -> iterative evaluation.
   This reproduces the original ARDA behaviour on genuinely relational
   (multi-table) data and is kept for completeness / comparison, but it is
   NOT the pipeline used for this project's own benchmarks.

Both modes share the same feature-selection engine (RIFS, see
feature_selection.py) and the same "add one more top-ranked feature and
retrain" evaluation loop, factored out into ``_iterative_feature_evaluation``
below so that model-training / CSV-logging logic exists exactly once.
"""

from __future__ import annotations

import os
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from typing import Any, Callable, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import paths
from .coreset import CoresetConstructor
from .joins import join_tables
from .feature_selection import RIFS, one_hot_encode_categorical, _is_classification
from .join_graph import JoinGraph
from .train_model import ModelTraining


def _to_original_feature(sel_col: str, original_cols: List[str]) -> str:
    """Map a one-hot encoded column name back to its original feature."""
    if sel_col in original_cols:
        return sel_col

    for orig in original_cols:
        if sel_col.startswith(orig + "_"):
            return orig

    return sel_col # Fallback


def _map_back_to_original(binarized_cols: List[str], original_cols: List[str]) -> List[str]:
    kept = {_to_original_feature(col, original_cols) for col in binarized_cols}
    return [c for c in original_cols if c in kept]


def map_ranking_to_original(ranking_df: pd.DataFrame, original_cols: List[str]) -> pd.DataFrame:
    df = ranking_df.copy()

    # Map encoded features to original features
    df["original_feature"] = df["feature"].apply(lambda col: _to_original_feature(col, original_cols))

    # Sort so best rows appear first
    df = df.sort_values(["original_feature", "r_star", "mean_agg_rank"], ascending=[True, False, True])

    # Keep best dummy per original feature
    df_best = df.groupby("original_feature", as_index=False).first()

    # Restore original feature names
    df_best["feature"] = df_best["original_feature"]
    df_best = df_best.drop(columns="original_feature")

    # Global ranking
    df_best = df_best.sort_values(["r_star", "mean_agg_rank"], ascending=[False, True]).reset_index(drop=True)
    df_best["rank"] = range(1, len(df_best) + 1)

    return df_best

# ---------------------------------------------------------------------------
# Candidate join descriptor (only used by the ORIGINAL ARDA / join-based path)
# ---------------------------------------------------------------------------

CandidateJoin = Dict[str, Any]
# Required keys: "table", "base_key", "foreign_key"
# Optional keys: "score" (priority), "suffix"

# A callable that, given the top-k ranked feature names for one iteration of
# the evaluation loop, returns the table to train/evaluate a model on.
SubsetBuilder = Callable[[List[str]], pd.DataFrame]


# ---------------------------------------------------------------------------
# ARDA
# ---------------------------------------------------------------------------

class ARDA:
    """
    Automatic Relational Data Augmentation.

    Parameters
    ----------
    target_col        : name of the label / target column in the base table
    coreset_size      : rows to keep for efficient join + feature selection
                        (None = use all rows). Only relevant to the original,
                        join-based pipeline (`fit_transform`).
    coreset_method    : 'uniform' | 'stratified' | 'sketch'
    task              : 'classification' | 'regression' | 'auto'
    rifs_eta          : RIFS noise fraction (default 0.2)
    rifs_k            : RIFS repetitions (default 10)
    rifs_nu           : RIFS RF weight in aggregate ranking (default 0.5)
    cv                : cross-validation folds for evaluation
    random_state      : int
    verbose           : bool
    """

    def __init__(self, target_col: str, coreset_size: Optional[int] = None, coreset_method: str = "uniform", task: str = "auto", metric: str = "accuracy",
        rifs_eta: float = 0.2, rifs_k: int = 10, rifs_nu: float = 0.5, cv: int = 3, random_state: int = 0, verbose: bool = True, benchmark: str = None):

        self.target_col = target_col
        self.coreset_size = coreset_size
        self.coreset_method = coreset_method
        self.task = task
        self.metric = metric
        self.rifs_eta = rifs_eta
        self.rifs_k = rifs_k
        self.rifs_nu = rifs_nu
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.benchmark = benchmark

        # Results populated by fit_transform / fit_transform_full_table
        self.selected_features_: Optional[List[str]] = None
        self.used_joins_: List[CandidateJoin] = []
        self.base_score_: Optional[float] = None
        self.augmented_score_: Optional[float] = None
        self._task_resolved: Optional[str] = None

    # ==================================================================
    # PRIMARY ENTRY POINT — used throughout this project's experiments
    # ==================================================================
    #
    # No join discovery or execution happens here: `dataset` is assumed to
    # already contain every candidate feature (this project's benchmarks in
    # data/ are pre-built, already-augmented tables). Only ARDA's
    # feature-selection stage (RIFS) plus iterative evaluation is run.
    # ==================================================================

    def fit_transform_full_table(self, dataset: pd.DataFrame, target_column: str, task: str, metric: str,
                                  num_features: Optional[int] = None, max_features: int = 40) -> pd.DataFrame:
        """
        Run RIFS feature selection directly on `dataset`, then train a
        model. Every step is logged to results.csv / results_regression.csv.

        This is the "feature-selection-only" mode of ARDA: unlike
        `fit_transform`, it performs no join discovery/execution
        (Section 4 of the original ARDA paper) — `dataset` is treated as a
        single, already-assembled table.

        Two modes, controlled by `num_features`:

        - `num_features` given (the default for the single-CSV "system",
          see pipeline.py): train exactly ONE model, using the top
          `num_features` ranked columns (capped to however many candidate
          columns actually exist). Returns a 1-row DataFrame. This is what
          a normal user wants — one model, not forty.
        - `num_features=None` (the default of this method itself, kept for
          the benchmarks.yaml reproduction sweep — see arda/run_benchmarks.py
          and experiments/run_paper_experiments.py): retrain the model once
          per feature count from 1 up to `max_features`, tracing out the
          accuracy/#features curve the paper's results are built from.
          Returns a DataFrame with one row per number-of-features tried.

        Parameters
        ----------
        dataset       : table containing `target_column` plus every candidate feature
        target_column : name of the label column
        task          : 'classification' | 'regression'
        metric        : evaluation metric passed through to ModelTraining
        num_features  : exact number of top-ranked features to train with.
                        If None (default), sweep 1..max_features instead.
        max_features  : cap on how many top-ranked features to try when
                        `num_features` is None (default 40). Ignored when
                        `num_features` is given.
        """
        self.target_col = target_column
        self._task_resolved = task
        candidate_cols = [c for c in dataset.columns if c != target_column]

        start_time = time.time()
        feature_ranking = self.select_features(dataset, candidate_cols)
        augmentation_time = time.time() - start_time
        print(feature_ranking)

        def build_subset(final_cols: List[str]) -> pd.DataFrame:
            return dataset[final_cols + [target_column]]

        if num_features is not None:
            if num_features < 1:
                raise ValueError(f"num_features must be >= 1, got {num_features}")
            capped = min(num_features, len(candidate_cols))
            if capped < num_features and self.verbose:
                print(f"Requested num_features={num_features}, but only {capped} candidate feature(s) "
                      f"are available -- using {capped}.")
            feature_counts = [capped]
        else:
            feature_counts = list(range(1, min(max_features, len(candidate_cols)) + 1))

        return self._iterative_feature_evaluation(
            feature_ranking=feature_ranking,
            build_subset=build_subset,
            target_column=target_column,
            task=task,
            metric=metric,
            feature_counts=feature_counts,
            augmentation_time=augmentation_time,
            presets="good_quality",
        )

    # ==================================================================
    # ORIGINAL ARDA — full pipeline (coreset + joins + feature selection)
    # ==================================================================
    #
    # Kept for completeness / to reproduce the original ARDA behaviour on
    # genuinely relational (multi-table) data. This project's own
    # experiments do NOT use this path — see `fit_transform_full_table`
    # above for the pipeline actually used.
    # ==================================================================

    def fit_transform(self, base_df: pd.DataFrame, candidate_joins: List[CandidateJoin],
                       num_features: Optional[int] = None) -> pd.DataFrame:
        """
        Run the full ARDA pipeline on `base_df` given `candidate_joins`:
        coreset construction -> join execution -> RIFS feature selection on
        the joined coreset -> iterative evaluation on the full (un-sampled)
        joined table.

        `num_features`: same meaning as on `fit_transform_full_table` --
        None (default) sweeps every feature count from 1 up to however many
        ranked features exist; an int trains exactly one model with that
        many top-ranked features (capped to however many exist).
        """
        logger.info("=== ARDA: Starting augmentation pipeline (coreset + joins + feature selection) ===")

        # 0. Resolve task (if needed)
        y_full = base_df[self.target_col]
        self._task_resolved = (
            self.task if self.task != "auto"
            else ("classification" if _is_classification(y_full.values) else "regression")
        )
        logger.info(f"Task: {self._task_resolved}")

        # 1. Coreset: joins + feature selection run on a cheap sample first
        coreset_size = self.coreset_size or len(base_df)
        constructor = CoresetConstructor(method=self.coreset_method, size=coreset_size, random_state=self.random_state)
        coreset = constructor.construct(base_df, target_col=self.target_col)
        print(f"Coreset: {len(coreset)} rows (full table: {len(base_df)})")

        # 2. Sort joins by score (if provided) and execute them on the coreset
        joins_to_use = sorted(candidate_joins, key=lambda j: j.get("score", 0), reverse=True)
        start_cols = set(coreset.columns)
        coreset_aug = self.execute_joins(coreset.copy(), joins_to_use)
        new_cols = [c for c in coreset_aug.columns if c not in start_cols]
        print(f"{len(new_cols)} new feature(s) after join")

        # 3. Feature selection on the joined coreset
        start_time = time.time()
        feature_ranking = self.select_features(coreset_aug, new_cols)
        augmentation_time = time.time() - start_time
        print(feature_ranking)

        # Join keys are structural (they exist to link tables), not real
        # features -- exclude them from the ranking before evaluation.
        join_keys: set = {j["base_key"] for j in joins_to_use}
        filtered_ranking = feature_ranking[
            feature_ranking["feature"].apply(lambda c: c not in join_keys)
        ].reset_index(drop=True)

        # 4. Execute the same joins once on the FULL (un-sampled) base table
        print("Applying selected augmentation to full table…")
        full_aug = self.augment_full_table(base_df, joins_to_use)
        orig_cols = list(base_df.columns)

        def build_subset(final_cols: List[str]) -> pd.DataFrame:
            extra = [c for c in final_cols if c not in orig_cols and c in full_aug.columns]
            subset = full_aug[orig_cols + extra]
            # Drop join-key / row-index artifact columns -- structural, not features
            keep = [c for c in subset.columns if "Key_" not in c and "idx." not in c]
            return subset[keep]

        print(f"Total time for augmentation -> {augmentation_time}")

        if num_features is not None:
            if num_features < 1:
                raise ValueError(f"num_features must be >= 1, got {num_features}")
            capped = min(num_features, len(filtered_ranking))
            feature_counts = [capped]
        else:
            feature_counts = list(range(1, len(filtered_ranking) + 1))

        return self._iterative_feature_evaluation(
            feature_ranking=filtered_ranking,
            build_subset=build_subset,
            target_column=self.target_col,
            task=self._task_resolved,
            metric=self.metric,
            feature_counts=feature_counts,
            augmentation_time=augmentation_time,
            presets="high_quality",
        )

    def fit_transform_from_connections(self, connections_path: str, base_table_name: str, table_dir: str = ".") -> pd.DataFrame:
        """
        Entry point that builds a JoinGraph from a connections CSV file and then runs the full ARDA pipeline.

        Parameters
        ----------
        connections_path  : path to connections.csv / connections.tsv
        base_table_name   : filename of the base table (e.g. 'table_0_0.csv')
        table_dir         : directory containing all table CSV files

        Returns the augmented base table as a pd.DataFrame.
        """
        graph = JoinGraph(connections_path=connections_path, table_dir=table_dir)
        print(graph.summary(base_table_name))

        base_df = graph.load_base_table(base_table_name)
        candidate_joins = graph.get_candidate_joins(base_table_name)
        print(f"Graph traversal: {len(candidate_joins)} reachable table(s)")

        return self.fit_transform(base_df, candidate_joins)

    def execute_joins(self, df: pd.DataFrame, join_list: List[CandidateJoin]) -> pd.DataFrame:
        """Execute `join_list` against `df` -- used on the coreset during join discovery."""
        return self._run_joins(df, join_list, context="coreset")

    def augment_full_table(self, base_df: pd.DataFrame, joins: List[CandidateJoin]) -> pd.DataFrame:
        """Execute `joins` against the full (un-sampled) base table."""
        return self._run_joins(base_df.copy(), joins, context="full table")

    def _run_joins(self, df: pd.DataFrame, join_list: List[CandidateJoin], context: str) -> pd.DataFrame:
        """Shared join-execution loop behind `execute_joins` / `augment_full_table`."""
        for j in tqdm(join_list, desc=f"Joining ({context})"):
            try:
                df = join_tables(base=df, base_key=j["base_key"],
                    foreign=j["table"], foreign_key=j["foreign_key"], suffix=j.get("suffix", "_aug"),
                )
            except Exception as exc:
                warnings.warn(f"Join failed ({context}) [{j.get('base_key')} → {j.get('foreign_key')}]: {exc}")
        return df

    # ==================================================================
    # Shared internals (used by BOTH entry points above)
    # ==================================================================

    def select_features(self, df: pd.DataFrame, new_cols: List[str]) -> pd.DataFrame:
        """Run RIFS over `new_cols` and return them ranked best-first."""
        if not new_cols:
            return []
        y = df[self.target_col]
        X_raw = df[[c for c in new_cols if c in df.columns]]

        # Encode any categorical columns before passing to selector, as we can only deal with numerical features
        X_bin = one_hot_encode_categorical(X_raw)
        X_num = X_bin.select_dtypes(include=[np.number])
        if X_num.empty:
            return []

        rifs = RIFS(eta=self.rifs_eta, k=self.rifs_k, nu=self.rifs_nu, task=self._task_resolved or self.task,
                    cv=self.cv, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranking_df = rifs.rank_features(X_num, y)
            print(map_ranking_to_original(ranking_df, new_cols).head(50))
        return map_ranking_to_original(ranking_df, new_cols)

    def _iterative_feature_evaluation(self, feature_ranking: pd.DataFrame, build_subset: SubsetBuilder, target_column: str, task: str,
        metric: str, feature_counts: List[int], augmentation_time: float, presets: str) -> pd.DataFrame:
        """
        Shared "take the top-N ranked features and train" loop.

        For each `num_features` in `feature_counts`: take the top
        `num_features` columns from `feature_ranking`, build the
        corresponding table via `build_subset(final_cols)`, train +
        evaluate a model on it, and log the result. This is the piece of
        logic that `fit_transform_full_table` (no joins) and
        `fit_transform` (with joins) both need, factored out so it only
        exists once.

        `feature_counts` is usually either a single value (one model, the
        "system" use case) or `range(1, max_features + 1)` (a full sweep,
        the reproducibility use case) -- the caller decides which.

        Parameters
        ----------
        feature_ranking : output of `select_features`, sorted best-first
        build_subset    : callable(final_cols) -> pd.DataFrame; returns the
                           table to train on for that feature subset
        feature_counts  : the exact list of "top-N features" values to try
        presets         : AutoGluon presets (e.g. 'good_quality', 'high_quality')

        Returns a DataFrame with one row per entry in `feature_counts`.
        """
        rows: List[Dict[str, Any]] = []

        for num_features_to_include in feature_counts:
            final_cols = feature_ranking["feature"].head(num_features_to_include).tolist()
            subset = build_subset(final_cols)

            metrics = ModelTraining(subset, target_column, 180, presets, task, metric).train_model_base()

            row = {
                "benchmark": self.benchmark, "problem": task, "augmentation_time": augmentation_time, **metrics,
                "num_new_features": num_features_to_include, "selected_features": ",".join(sorted(final_cols)),
            }
            self._log_result(row, task)
            rows.append(row)

        return pd.DataFrame(rows)

    def _log_result(self, row: Dict[str, Any], task: str) -> None:
        """Append one evaluation result to results/results_regression.csv (regression) or
        results/results.csv (classification) -- a running log of every individual model
        fit, independent of and in addition to whatever the caller does with the
        returned DataFrame (e.g. run_paper_experiments.py's own results/reproduced_*.csv).

        Writes under paths.RESULTS_DIR, not a bare cwd-relative filename -- found by
        smoke-testing run_paper_experiments.py, which left a stray results_regression.csv
        at the repo root because this method used to write to cwd directly, the one
        remaining hardcoded-relative-path spot in the package (everything else already
        goes through paths.py for exactly this reason)."""
        csv_path = paths.RESULTS_DIR / ("results_regression.csv" if task == "regression" else "results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        file_exists = csv_path.exists()
        df.to_csv(csv_path, mode="a" if file_exists else "w", header=not file_exists, index=False)
