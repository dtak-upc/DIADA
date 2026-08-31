"""
Coreset Construction

Two strategies:
  - uniform    : simple random sampling
  - stratified : proportional sampling per label categories (classification)

IDEALLY, coresets should not be used; and the full table should be processed instead.
DEFAULT (in case they are used): uniform for regression tasks and stratified for classification tasks
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging
logger = logging.getLogger(__name__)


class CoresetConstructor:
    """
    Builds a representative subset of the base table to reduce the cost of joining, feature selection and model training.

    Parameters
    ----------
    method : {'uniform', 'stratified'}
    size   : int or None
        Number of rows to keep. If None or >= len(df), the full table is returned unchanged.
    random_state : int
    """

    def __init__(self, method: str = "uniform", size: Optional[int] = None, random_state: int = 0):
        if method not in ("uniform", "stratified", "sketch"):
            raise ValueError(f"Unknown coreset method '{method}'")
        self.method = method
        self.size = size
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)


    def construct(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Return a coreset of dataframe `df`.

        Parameters
        ----------
        df         : input DataFrame (base table or already-joined table)
        target_col : name of the label column (required for stratified/sketch)
        """
        if self.size is None or self.size >= len(df):
            return df.copy()

        dispatch = {"uniform": self._uniform, "stratified": self._stratified}
        return dispatch[self.method](df, target_col)

    # Method 1: Uniform sampling
    def _uniform(self, df: pd.DataFrame, _target_col) -> pd.DataFrame:
        return df.sample(n=self.size, random_state=self.random_state).reset_index(drop=True)

    # Method 2: Stratified sampling
    def _stratified(self, df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
        if target_col is None or target_col not in df.columns:
            logger.warning(f"Target column ({target_col}) is None or it does not appear in the df: {df.columns.to_list()}")
            logger.warning(f"Falling back to uniform sampling")
            return self._uniform(df, target_col)

        classes = df[target_col].unique()
        n_classes = len(classes)
        base_per_class = max(1, self.size // n_classes)

        parts = []
        for cls in classes:
            cls_df = df[df[target_col] == cls]
            n = min(base_per_class, len(cls_df))
            parts.append(cls_df.sample(n=n, random_state=self.random_state))

        result = pd.concat(parts)

        # Top up to self.size with uniform random from the remainder
        deficit = self.size - len(result)
        if deficit > 0:
            remaining = df.drop(index=result.index)
            extra_n = min(deficit, len(remaining))
            if extra_n > 0:
                result = pd.concat([result, remaining.sample(n=extra_n, random_state=self.random_state)])

        return result.reset_index(drop=True)