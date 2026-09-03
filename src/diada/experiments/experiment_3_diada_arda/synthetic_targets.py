import numpy as np
import pandas as pd

from typing import Literal

from scipy.stats import rankdata
from scipy.special import expit  # stable sigmoid

def build_synthetic_target(df: pd.DataFrame, target_col: str, task_type: Literal["binary", "multiclass", "regression"], 
                            source_features: list[str], rng: np.random.Generator) -> np.ndarray | pd.Categorical:
    """
    Strategy
    --------
    1. Encode every source feature numerically (ordinal for categoricals).
    2. Compute a weighted linear score  s = X @ w  where w ~ N(0,1).
       The score captures correlations between a random linear combination of
       the source features.  Non-linear tasks benefit from a rank-normalisation
       step that makes the score uniform, which is then re-scaled.
    3. Transform the score into the appropriate output type:
       - regression  → shift/scale to match the original target's mean/std, then
                       add Gaussian noise (SNR ≈ 4  →  R² ≈ 0.80-0.90)
       - binary      → pass score through sigmoid → Bernoulli sample
                       (decision boundary shift ± 0.3 from 0.5 to add variation)
       - multiclass  → split score into K quantile buckets where K = n_classes,
                       with a small probability of flipping to a neighbour class
    """
    X = encode_features_numeric(df, source_features, rng)

    # Random weight vector -> normalised so score variance is ~1
    w = rng.standard_normal(X.shape[1])
    score = X @ w
    score_std = np.std(score)
    if score_std > 0:
        score = score / score_std  # z-score the linear combination

    # Rank-normalise to make the score robust to outlier features
    ranks = rankdata(score, method="average")
    score_ranked = (ranks - 1) / (len(ranks) - 1 + 1e-9)

    if task_type == "regression":
        return synthetic_regression(df[target_col], score_ranked, rng)
    elif task_type == "binary":
        return synthetic_binary(df[target_col], score_ranked, rng)
    else:
        return synthetic_multiclass(df[target_col], score_ranked, rng)
    

def encode_features_numeric(df: pd.DataFrame, feature_cols: list[str], rng: np.random.Generator) -> np.ndarray:
    """
    Return a 2-D float array (n × p) encoding `feature_cols`.

    - Numeric columns: standard-scaled (z-score), NaNs filled with 0.
    - Categorical / object columns: ordinal-encoded by sorted category index,
      then standard-scaled. Unknown / NaN → 0.
    """
    parts: list[np.ndarray] = []

    for col in feature_cols:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            vals = series.to_numpy(dtype=float, na_value=np.nan)
            vals = np.where(np.isnan(vals), 0.0, vals)
        else: # Ordinal encode: sort unique categories alphabetically → integer index
            categories = sorted(series.dropna().unique().tolist(), key=str)
            cat_map = {cat: i for i, cat in enumerate(categories)}
            # Cast to object first: if `series` is a pandas Categorical, .map() would
            # return another Categorical whose categories are the mapped integers,
            # and .fillna(-1) then fails because -1 isn't one of those categories.
            mapped = series.astype(object).map(cat_map)
            vals = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=float)
            vals = np.where(np.isnan(vals), -1.0, vals)

        std = np.std(vals) # Standard-scale
        if std > 0:
            vals = (vals - np.mean(vals)) / std

        parts.append(vals)

    if not parts: # Fallback: return noise if no usable features
        return rng.standard_normal((len(df), 1))

    return np.column_stack(parts)
    

def synthetic_regression(y_orig: pd.Series, score_uniform: np.ndarray, rng: np.random.Generator, snr: float = 4.0) -> np.ndarray:
    """
    Regression synthetic target.

    Steps:
      1. Re-scale the uniform score to match the original target's mean/std.
         (score_uniform ∈ [0,1] → stretch to the original range)
      2. Add Gaussian noise scaled so that Signal-to-Noise Ratio ≈ `snr`.
         SNR = var(signal) / var(noise)  → expected R² ≈ snr / (snr + 1)
         With snr=4 → R² ≈ 0.80.

    We deliberately avoid copying the original target directly; instead we
    derive a shifted/scaled version from the score, which shares structure
    with the source features but is not identical to the true target.
    """
    y = y_orig.to_numpy(dtype=float)
    y_mean, y_std = np.nanmean(y), np.nanstd(y)
    if y_std == 0:
        y_std = 1.0

    # Map score ∈ [0,1] to approximately the same range as the original target
    # Use a small shift (±0.5 std) to make the synthetic target distinct
    shift = rng.uniform(-0.5, 0.5) * y_std
    signal = score_uniform * (3 * y_std) + (y_mean - 1.5 * y_std) + shift

    signal_std = np.std(signal)
    if signal_std == 0:
        signal_std = 1.0
    noise_std = signal_std / np.sqrt(snr)
    noise = rng.normal(0, noise_std, size=len(signal))

    return signal + noise


def synthetic_binary(y_orig: pd.Series, score_uniform: np.ndarray, rng: np.random.Generator, 
                     base_threshold: float = 0.5, threshold_jitter: float = 0.15, ) -> np.ndarray:
    """
    Binary classification synthetic target.

    The uniform score is treated as a latent probability.  A threshold
    (slightly perturbed from 0.5 to vary the class balance) converts it into
    {0, 1} labels – or the original class labels if they are not numeric.

    The score is passed through a sigmoid with a steepness parameter β so
    that the resulting probabilities are not too close to 0.5 everywhere
    (which would make the task trivially hard).  β ∈ [3, 6] gives a good
    learnable range.
    """
    classes = y_orig.dropna().unique()
    try: # Sort for determinism: put 0/False/lower value first
        classes = sorted(classes)
    except TypeError:
        classes = sorted(classes, key=str)

    # Logit-transform score from [0,1] → real line, then apply sigmoid with β
    epsilon = 1e-6
    latent = np.log(score_uniform + epsilon) - np.log(1 - score_uniform + epsilon)
    beta = rng.uniform(3.0, 6.0)
    prob = expit(beta * latent)

    # Threshold with a small random shift to vary class balance slightly
    threshold = base_threshold + rng.uniform(-threshold_jitter, threshold_jitter)
    labels_int = (prob >= threshold).astype(int)

    # Map back to original label dtype
    label_map = {0: classes[0], 1: classes[-1]}
    return np.array([label_map[v] for v in labels_int])


def synthetic_multiclass(y_orig: pd.Series, score_uniform: np.ndarray, rng: np.random.Generator, flip_prob: float = 0.08) -> np.ndarray:
    """
    Multiclass classification synthetic target.

    The uniform score is quantile-binned into K buckets (K = number of
    original classes).  The bucket index maps to a class label.  A small
    fraction (`flip_prob`) of labels are randomly replaced by an adjacent
    class to add realistic confusion without destroying learnability.

    The class ordering follows the original target's sorted unique values,
    so the synthetic target preserves ordinal structure when it exists.
    """
    classes = y_orig.dropna().unique()
    try:
        classes = sorted(classes)
    except TypeError:
        classes = sorted(classes, key=str)
    K = len(classes)

    if K < 2: # Degenerate: return the only class
        return np.full(len(score_uniform), classes[0])

    # Assign class by quantile bucket
    boundaries = np.linspace(0, 1, K + 1)
    class_indices = np.searchsorted(boundaries[1:], score_uniform, side="left")
    class_indices = np.clip(class_indices, 0, K - 1)

    # Random adjacent-class flips to add noise without large errors
    flip_mask = rng.random(len(class_indices)) < flip_prob
    n_flips = flip_mask.sum()
    if n_flips > 0:
        # Each flip moves ±1 class index (clipped to valid range)
        deltas = rng.choice([-1, 1], size=n_flips)
        class_indices[flip_mask] = np.clip(class_indices[flip_mask] + deltas, 0, K - 1)

    return np.array([classes[i] for i in class_indices])