"""
ARDA Feature Selection (Sections 5–6)

Core algorithm: Random Injection Feature Selection (RIFS)
  - Algorithm 1: feature selection via random injection
  - Algorithm 2: random feature injection subroutine (moment-matching)
  - Algorithm 3: wrapper algorithm (threshold search)

The ℓ2,1-norm minimisation from Eq. (1) is approximated by sklearn Lasso
(single-output) / MultiTaskLasso (multi-output).  The paper's experiments
show the combination of RF + SR in RIFS is what drives performance; the exact
solver for the SR component matters less.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Categorical encoding (i.e. to handle categorical data)
# ---------------------------------------------------------------------------

def one_hot_encode_categorical(df: pd.DataFrame, max_cardinality: int = 20, drop_original: bool = True) -> pd.DataFrame:
    """
    One-hot encode categorical columns so they can be used in numeric feature selectors.
    Columns with cardinality > max_cardinality are dropped (too many dummies would dominate the feature space).

    Parameters
    ----------
    df               : DataFrame that may contain object/category columns
    max_cardinality  : columns with more unique values than this are dropped
    drop_original    : if True, remove the original string column after encoding

    Returns a new DataFrame with object columns replaced by 0/1 dummies.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > max_cardinality:
            df = df.drop(columns=[col])
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        if drop_original:
            df = df.drop(columns=[col])

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_classification(y: np.ndarray) -> bool:
    return y.dtype == object or len(np.unique(y)) <= 20


def _encode_labels(y: np.ndarray) -> np.ndarray:
    if y.dtype == object:
        le = LabelEncoder()
        return le.fit_transform(y)
    return y.astype(float)


def _get_rf(task: str, n_estimators: int = 100, random_state: int = 0):
    if task == "classification":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def rf_importances(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    rf = _get_rf(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y)
    return rf.feature_importances_


def lasso_importances(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """Approximate ℓ2,1-norm SR via Lasso; return |coef| as importance."""
    y_enc = _encode_labels(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LassoCV(cv=3, max_iter=5000, random_state=0, n_jobs=-1)
        model.fit(X, y_enc)
    return np.abs(model.coef_)


def rank_from_importances(importances: np.ndarray) -> np.ndarray:
    """Return rank array (0 = most important)."""
    order = np.argsort(importances)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


# ---------------------------------------------------------------------------
# Algorithm 2 – Random Feature Injection Subroutine
# ---------------------------------------------------------------------------

def inject_random_features(X: np.ndarray, eta: float = 0.2, random_state: int = None) -> Tuple[np.ndarray, int]:
    """
    Generate ηd random features whose marginal distribution matches the columns of X.

    Returns
    -------
    X_aug : X with random features appended as extra columns
    t     : number of injected features
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    t = max(1, int(np.ceil(eta * d)))

    try:
        col_mu = np.mean(X, axis=0)        # (d,)
        col_std = np.std(X, axis=0) + 1e-8
        noise_cols = rng.standard_normal((n, t))
        for i in range(t):
            ref_idx = rng.integers(0, d)
            noise_cols[:, i] = noise_cols[:, i] * col_std[ref_idx] + col_mu[ref_idx]
    except Exception:  # Fallback: standard normal
        noise_cols = rng.standard_normal((n, t))

    X_aug = np.hstack([X, noise_cols])
    return X_aug, t


# ---------------------------------------------------------------------------
# Algorithm 1 – Single RIFS run (with detailed outputs)
# ---------------------------------------------------------------------------

def rifs_single_run_details(X: np.ndarray, y: np.ndarray, task: str, eta: float = 0.2, nu: float = 0.5, random_state: int = 0) -> dict:
    """
    One iteration of Algorithm 1. Returns a dict with:
    - 'scores'   : binary array — 1 if feature beats all injected noise
    - 'rf_rank', 'sr_rank', 'agg_rank' : rank arrays (lower = better)
    - 'rf_imp', 'sr_imp'               : raw importance arrays
    All arrays contain only the original d features (injected columns excluded).
    """
    d = X.shape[1]
    X_aug, _ = inject_random_features(X, eta=eta, random_state=random_state)

    rf_imp = rf_importances(X_aug, y, task)
    rf_rank = rank_from_importances(rf_imp)

    try:
        sr_imp = lasso_importances(X_aug, y, task)
        sr_rank = rank_from_importances(sr_imp)
    except Exception:  # Fallback to RF if SR fails
        sr_imp = rf_imp.copy()
        sr_rank = rf_rank.copy()

    agg_rank = nu * rf_rank + (1 - nu) * sr_rank

    random_feature_ranks = agg_rank[d:]
    max_random_rank = np.max(random_feature_ranks) if len(random_feature_ranks) > 0 else np.inf
    scores = (agg_rank[:d] < max_random_rank).astype(float)

    return {
        "scores":   scores,
        "rf_rank":  rf_rank[:d].astype(float),
        "sr_rank":  sr_rank[:d].astype(float),
        "agg_rank": agg_rank[:d].astype(float),
        "rf_imp":   rf_imp[:d].astype(float),
        "sr_imp":   sr_imp[:d].astype(float),
    }


def rifs_single_run(X: np.ndarray, y: np.ndarray, task: str, eta: float = 0.2, nu: float = 0.5, random_state: int = 0) -> np.ndarray:
    """Thin wrapper — returns only the scores vector from a single RIFS run."""
    return rifs_single_run_details(X, y, task, eta=eta, nu=nu, random_state=random_state)["scores"]


# ---------------------------------------------------------------------------
# RIFS – Full Algorithm
# ---------------------------------------------------------------------------

class RIFS:
    """
    Random Injection Feature Selection.

    Parameters
    ----------
    eta          : fraction of features to inject as random noise (default 0.2)
    k            : number of repetitions (default 10)
    nu           : RF weight in aggregate ranking; SR weight = 1-nu (default 0.5)
    thresholds   : sequence of τ values for the wrapper (Algorithm 3)
    task         : 'classification' | 'regression' | 'auto'
    cv           : cross-validation folds for wrapper evaluation
    random_state : int
    """

    def __init__(self, eta: float = 0.2, k: int = 10, nu: float = 0.5, thresholds: Optional[Sequence[float]] = None, task: str = "auto",
        cv: int = 3, random_state: int = 42, max_cardinality: int = 20):

        self.eta = eta
        self.k = k
        self.nu = nu
        self.thresholds = thresholds if thresholds is not None else [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        self.task = task
        self.cv = cv
        self.random_state = random_state
        self.max_cardinality = max_cardinality

        self.selected_features_: Optional[np.ndarray] = None
        self.r_star_: Optional[np.ndarray] = None
        self.best_tau_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_best_feature_subset(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit RIFS and return the best feature subset as a DataFrame."""
        X_arr, y_arr, feature_names = self.prepare(X, y)
        task = self.resolve_task(y_arr)

        # 1. Compute r_star scores
        r_star, _ = self.accumulate_runs(X_arr, y_arr, task)
        self.r_star_ = r_star

        # 2. Obtain the best feature subset
        best_tau, best_cols = self.wrapper(X_arr, y_arr, task, r_star, feature_names)
        self.best_tau_ = best_tau
        self.selected_features_ = np.array(best_cols)

        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols]

    def rank_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compute an ordered ranking DataFrame of the original features.

        Returns a DataFrame with columns:
          - feature                      : feature name
          - r_star                       : fraction of runs where the feature beat all injected noise
          - mean_agg_rank, std_agg_rank  : mean / std of aggregate rank across runs (lower = better)
          - mean_rf_rank, mean_sr_rank   : mean ranks for RF and SR components
          - mean_rf_imp, mean_sr_imp     : mean raw importances
        Sorted by r_star descending, then mean_agg_rank ascending.
        """
        X_arr, y_arr, feature_names = self.prepare(X, y)
        task = self.resolve_task(y_arr)

        r_star, acc = self.accumulate_runs(X_arr, y_arr, task)

        df = pd.DataFrame({
            "feature":      feature_names,
            "r_star":       r_star,
            "mean_agg_rank": np.mean(acc["agg_rank"], axis=0),
            "std_agg_rank":  np.std(acc["agg_rank"],  axis=0),
            "mean_rf_rank":  np.mean(acc["rf_rank"],  axis=0),
            "mean_sr_rank":  np.mean(acc["sr_rank"],  axis=0),
            "mean_rf_imp":   np.mean(acc["rf_imp"],   axis=0),
            "mean_sr_imp":   np.mean(acc["sr_imp"],   axis=0),
        })
        return df.sort_values(["r_star", "mean_agg_rank"], ascending=[False, True]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def prepare(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_bin = one_hot_encode_categorical(X)
        X_num = X_bin.select_dtypes(include=[np.number])
        feature_names = list(X_num.columns)
        X_arr = X_num.fillna(X_num.median()).values.astype(float)
        return X_arr, y.values, feature_names

    def resolve_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            return self.task
        return "classification" if _is_classification(y) else "regression"

    def accumulate_runs(self, X: np.ndarray, y: np.ndarray, task: str) -> Tuple[np.ndarray, dict]:
        """
        Run RIFS k times and accumulate per-run statistics.

        Returns
        -------
        r_star : (d,) array — fraction of runs each feature beat all noise
        acc    : dict of (k, d) arrays keyed by 'agg_rank', 'rf_rank',
                 'sr_rank', 'rf_imp', 'sr_imp'
        """
        d = X.shape[1]
        y_enc = _encode_labels(y)
        keys = ["agg_rank", "rf_rank", "sr_rank", "rf_imp", "sr_imp"]
        acc = {key: np.zeros((self.k, d)) for key in keys}
        scores_sum = np.zeros(d)

        for run in tqdm(range(self.k), desc="Runs"):
            details = rifs_single_run_details(X, y_enc, task, eta=self.eta, nu=self.nu, random_state=self.random_state + run)
            scores_sum += details["scores"]
            for key in keys:
                acc[key][run] = details[key]

        return scores_sum / self.k, acc

    def score_subset(self, X: np.ndarray, y: np.ndarray, task: str, col_indices: List[int]) -> float:
        if len(col_indices) == 0:
            return -np.inf
        X_sub = X[:, col_indices]
        y_enc = _encode_labels(y)
        rf = _get_rf(task, n_estimators=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(rf, X_sub, y_enc, cv=self.cv, n_jobs=-1)
        return float(np.mean(scores))

    def wrapper(self, X: np.ndarray, y: np.ndarray, task: str, r_star: np.ndarray, feature_names: List[str]) -> Tuple[float, List[str]]:
        """Algorithm 3: search thresholds τ; stop when accuracy no longer improves."""
        ranked = np.argsort(r_star)[::-1]
        best_score = -np.inf
        best_tau = self.thresholds[0]
        best_cols: List[str] = feature_names  # Fallback: all features
        prev_score = -np.inf

        for tau in sorted(self.thresholds):
            sel_idx = [i for i, v in enumerate(r_star) if v >= tau]
            if len(sel_idx) == 0:
                break
            score = self.score_subset(X, y, task, sel_idx)
            if score > best_score:
                best_score = score
                best_tau = tau
                best_cols = [feature_names[i] for i in sel_idx]
            if score < prev_score:
                break
            prev_score = score

        # Exponential search over the sorted ranking (Sec 6.3 / Alg 3 note)
        n_features = len(feature_names)
        k = 1
        exp_best_score = -np.inf
        exp_best_idx: List[int] = list(ranked[:1])

        while k <= n_features:
            sel = list(ranked[:k])
            score = self.score_subset(X, y, task, sel)
            if score > exp_best_score:
                exp_best_score = score
                exp_best_idx = sel
            elif score < exp_best_score:
                lo, hi = k // 2, k
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    s = self.score_subset(X, y, task, list(ranked[:mid]))
                    if s >= exp_best_score:
                        exp_best_score = s
                        exp_best_idx = list(ranked[:mid])
                        lo = mid
                    else:
                        hi = mid
                break
            k = min(k * 2, n_features) if k * 2 <= n_features else n_features + 1

        if exp_best_score > best_score:
            best_cols = [feature_names[i] for i in exp_best_idx]

        return best_tau, best_cols or feature_names