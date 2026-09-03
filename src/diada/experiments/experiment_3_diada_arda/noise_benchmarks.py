from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Literal

from .synthetic_targets import build_synthetic_target
from .noise import random_numeric_noise, random_categorical_noise, shuffle_column
from .spurious_cluster import build_spurious_join_cluster


def generate_benchmark_noise_un(df: pd.DataFrame, target_col: str, random_state: int = 0):
    return generate_benchmark_noise(
        df, target_col,
        include_synthetic_targets=False,
        spurious_cluster_multiplier=0,
        random_state=random_state,
    )

def generate_benchmark_noise_un_synth_targets(df: pd.DataFrame, target_col: str, random_state: int = 0):
    return generate_benchmark_noise(
        df, target_col,
        include_synthetic_targets=True,
        spurious_cluster_multiplier=0,
        random_state=random_state,
    )

def generate_benchmark_noise_mn(noise_un_df: pd.DataFrame, report_un: dict,
                                 spurious_cluster_multiplier: float = 0.75, random_state: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Build the MN (multivariate-noise) variant by taking the already-generated
    UN (univariate-noise) dataframe (base features + noise columns produced
    for UN) and appending a spurious correlated cluster on top of it. This
    guarantees MN == UN + spurious cluster, rather than an independently
    regenerated (and therefore different) noisy dataset.
    """
    rng = np.random.default_rng(random_state)

    n_rows = len(noise_un_df)
    n_features = report_un["n_features"]
    n_spurious = int(round(spurious_cluster_multiplier * n_features))

    spurious_df = build_spurious_join_cluster(n_rows=n_rows, n_cols=n_spurious, rng=rng)

    df_out = pd.concat(
        [noise_un_df.reset_index(drop=True), spurious_df.reset_index(drop=True)],
        axis=1,
    )

    report = dict(report_un)
    report["spurious_cols"] = list(spurious_df.columns)
    report["total_new_columns"] = report_un["total_new_columns"] + len(spurious_df.columns)

    return df_out, report


# Note that this are per COLUMN. That is, if n_numeric_noise_features = 2, 2 numeric noise columns will be created FOR EACH of the original
# features (non-target). If we had 20 non-target features, we would have 40 numerical noise columns.
def generate_benchmark_noise(df: pd.DataFrame, target_col: str, n_numeric_noise_features: int = 2, n_categorical_noise_features: int = 2,
                       n_shuffled_features: int = 2, include_synthetic_targets: bool = False, spurious_cluster_multiplier: float = 0.75,
                       random_state: int = 0) -> tuple[pd.DataFrame, dict]:
    # Local RNG seeded from random_state -> output depends only on inputs, not on
    # how many times other functions have been called before this one.
    rng = np.random.default_rng(random_state)

    n_rows = len(df)
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe.")

    df_out = df.copy()
    new_cols: dict[str, np.ndarray | pd.Series | pd.Categorical] = {} # Accumulate new columns here, concat once at the end

    # 0. Detect task type
    task_type = detect_task_type(df_out[target_col])

    # 1. Pure-noise features
    noise_cols: list[str] = []

    # The generation of noise takes "variants" into account. E.g., noise numeric columns can have different distributions: normal, exponential, etc.
    for i in range(n_numeric_noise_features):
        col_name = f"noise_numeric_{i + 1}"
        new_cols[col_name] = random_numeric_noise(n_rows, variant=i, rng=rng)
        noise_cols.append(col_name)

    for i in range(n_categorical_noise_features):
        col_name = f"noise_categorical_{i + 1}"
        new_cols[col_name] = random_categorical_noise(n_rows, variant=i, rng=rng)
        noise_cols.append(col_name)

    # 2. Shuffled features
    shuffled_cols: list[str] = []

    for feat in feature_cols:
        orig = df_out[feat].values.copy()

        # Full reshuffle -> corruption_rate = 1
        for i in range(n_shuffled_features):
            col_name = f"{feat}_shuffled_{i + 1}"
            new_cols[col_name] = shuffle_column(orig, corruption_rate=1, rng=rng)
            shuffled_cols.append(col_name)

    # 3. Synthetic target columns
    # synthetic_target_1 is driven by the FIRST half of feature_cols
    # synthetic_target_2 is driven by the SECOND half of feature_cols
    split = max(1, n_features // 2)
    feature_halves = [feature_cols[:split], feature_cols[split:]]

    synthetic_target_cols: list[str] = []
    synthetic_target_source_features: dict[str, list[str]] = {}

    for target_idx, source_features in enumerate(feature_halves, start=1):
        col_name = f"synthetic_target_{target_idx}"

        synthetic_col = build_synthetic_target(df=df_out, target_col=target_col, task_type=task_type, source_features=source_features, rng=rng)

        if include_synthetic_targets:
            new_cols[col_name] = synthetic_col
            synthetic_target_cols.append(col_name)
            synthetic_target_source_features[col_name] = list(source_features)


    # 4. Spurious correlated cluster
    if spurious_cluster_multiplier > 0:
        n_spurious = int(round(spurious_cluster_multiplier * n_features))
        spurious_df = build_spurious_join_cluster(n_rows=n_rows, n_cols=n_spurious, rng=rng)

        for c in spurious_df.columns:
            new_cols[c] = spurious_df[c]

    # Concat all new columns at once to avoid DataFrame fragmentation
    df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    report = {
        "task_type": task_type,
        "n_features": n_features,
        "noise_cols": noise_cols,
        "heavily_shuffled_cols": shuffled_cols,
        "synthetic_target_cols": synthetic_target_cols,
        "synthetic_target_source_features": synthetic_target_source_features,
        "total_new_columns": (len(noise_cols) + len(shuffled_cols) + len(synthetic_target_cols))
    }

    return df_out, report


def detect_task_type(y: pd.Series) -> Literal["binary", "multiclass", "regression"]:
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        if n_unique == 2:
            return "binary"
        if n_unique <= 20 and pd.api.types.is_integer_dtype(y): # Heuristic: ≤20 unique integers → multiclass
            return "multiclass"
        return "regression"
    else:
        n_unique = y.nunique()
        if n_unique == 2:
            return "binary"
        return "multiclass"