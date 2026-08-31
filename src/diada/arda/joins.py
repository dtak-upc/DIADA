"""
ARDA Join Execution (Section 4)

ONLY HARD JOINS
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# Imputation
def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else "MISSING"
                df[col] = df[col].fillna(fill)
    return df


# Pre-aggregation helper (handles 1-to-many / many-to-many)
def _aggregate_foreign(foreign: pd.DataFrame, join_col: str) -> pd.DataFrame:
    """
    Aggregate a foreign table on `join_col` so each key appears exactly once.
    Numeric columns → mean; categorical columns → mode.
    """
    numeric_cols = foreign.select_dtypes(include=['number']).columns.tolist()
    string_cols = foreign.select_dtypes(include=['object', 'string']).columns.tolist()

    # Remove the key col from the columns to aggregate
    numeric_cols.remove(join_col) if join_col in numeric_cols else None
    string_cols.remove(join_col) if join_col in string_cols else None

    # Numeric aggregation (fast)
    df_num = foreign[[join_col] + numeric_cols].groupby(join_col, as_index=False).agg("mean")

    int_cols = foreign[numeric_cols].select_dtypes(include="int").columns
    for col in int_cols: # Integer columns have been transformed into double, so we cast them back to integer
        if (df_num[col] % 1 == 0).all():
            df_num[col] = df_num[col].astype("int64")

    # Categorical aggregation (optimized)
    cat_results = []

    for col in string_cols:
        cat = foreign[col].astype("category")
        codes = cat.cat.codes.to_numpy()
        uniques = cat.cat.categories

        tmp = pd.DataFrame({join_col: foreign[join_col].to_numpy(), "_code": codes})
        tmp = tmp[tmp["_code"] >= 0]  # drop NaNs

        # Count frequency per group
        counts = tmp.groupby([join_col, "_code"]).size().reset_index(name="cnt")

        # Select the code with highest count per group
        idx = counts.groupby(join_col)["cnt"].idxmax()
        modes = counts.loc[idx, [join_col, "_code"]]

        # Map back to original category
        modes[col] = uniques.take(modes["_code"].to_numpy())
        modes = modes.drop(columns="_code")

        cat_results.append(modes)

    # Merge numeric and categorical results
    df_agg = df_num
    for df_cat in cat_results:
        df_agg = df_agg.merge(df_cat, on=join_col, how="left")

    return df_agg


    # grp = foreign.groupby(join_col, sort=False)
    # agg_dict: Dict[str, Any] = {}
    # for col in foreign.columns:
    #     if col == join_col:
    #         continue
    #     if pd.api.types.is_numeric_dtype(foreign[col]):
    #         agg_dict[col] = "mean"
    #     else:
    #         agg_dict[col] = lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    # return grp.agg(agg_dict).reset_index()



# Public join function
def join_tables(base: pd.DataFrame, foreign: pd.DataFrame, base_key: str, foreign_key: str,
                suffix: str = "_aug") -> pd.DataFrame:
    """
    LEFT JOIN `foreign` onto `base` on (base_key, foreign_key).

    Parameters
    ----------
    base        : base table
    foreign     : candidate table to join
    base_key    : join column in base table
    foreign_key : join column in foreign table
    tolerance   : for nearest join, max allowed distance (None = no limit)
    suffix      : suffix appended to new column names to avoid clashes
    """
    base = base.copy()
    foreign_copy = foreign.copy()

    # Rename foreign columns to avoid clashes (keep join key unchanged)
    rename_map = {
        c: c + suffix
        for c in foreign.columns
        if c != foreign_key
        if c != foreign_key and (c + suffix) not in base.columns and c in base.columns
    }
    # print(rename_map)
    # print(foreign_copy.columns.to_list())
    foreign_copy = foreign_copy.rename(columns=rename_map)

    result = _hard_join(base, foreign_copy, base_key, foreign_key)
    return impute(result)


# Hard JOIN
def _hard_join(base: pd.DataFrame, foreign_copy: pd.DataFrame, base_key: str, foreign_key: str) -> pd.DataFrame:
    # Pre-aggregate to avoid 1-to-many
    foreign_copy = foreign_copy.dropna(axis=1, how='all')
    foreign_agg = _aggregate_foreign(foreign_copy, foreign_key)
    
    merged = base.merge(foreign_agg, left_on=base_key, right_on=foreign_key, how="left")

    # Drop duplicate key column from foreign if different name
    if foreign_key != base_key and foreign_key in merged.columns:
        merged = merged.drop(columns=[foreign_key])
    return merged.reset_index(drop=True)
