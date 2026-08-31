from typing import Dict, Optional

import numpy as np
import pandas as pd


def _infer_dtype(col: str) -> str:
    """Map column name keyword to pandas dtype."""
    if "Integer" in col:
        return "int64"
    if "Double" in col:
        return "float64"
    return "category"


class Dataset:
    """Wraps a pandas DataFrame loaded from a CSV, inferring each column's
    dtype from a keyword in its name ("Integer" -> int64, "Double" ->
    float64, anything else -> category).
    """

    def __init__(self, file_path: str, **kwargs):
        header_kwargs = dict(kwargs)
        header_kwargs["nrows"] = 0
        self.columns = pd.read_csv(file_path, **header_kwargs).columns.tolist()
        self.dtypes = {col: _infer_dtype(col) for col in self.columns}

        read_kwargs = dict(kwargs)
        read_kwargs["dtype"] = self.dtypes
        self.df = pd.read_csv(file_path, **read_kwargs)

    @classmethod
    def _fromDataFrame(cls, df: pd.DataFrame, dtypes: Dict[str, str]) -> "Dataset":
        """Build a Dataset directly from an already-constructed DataFrame,
        bypassing the CSV-reading constructor."""
        obj = cls.__new__(cls)
        obj.columns = df.columns.tolist()
        obj.dtypes = dtypes
        obj.df = df
        return obj

    def expand(self, n: int, m: int) -> "Dataset":
        """Build a new Dataset by sampling n rows and m columns from this
        one, with repetition - so n and/or m can exceed this dataset's own
        row/column count, to synthesize a larger dataset from a smaller one.
        """
        row_idx = np.random.randint(0, len(self.df), size=n)
        col_idx = np.random.randint(0, len(self.columns), size=m)

        new_df = self.df.iloc[row_idx, col_idx].reset_index(drop=True)
        # column names must stay unique even if the same original column was
        # sampled more than once, so suffix each with its position
        new_columns = [f"{self.columns[c]}_{i}" for i, c in enumerate(col_idx)]
        new_df.columns = new_columns

        new_dtypes = {
            new_col: self.dtypes[self.columns[c]] for new_col, c in zip(new_columns, col_idx)
        }

        return Dataset._fromDataFrame(new_df, new_dtypes)
