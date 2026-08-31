import numpy as np
import pandas as pd

def build_spurious_join_cluster(n_rows: int, n_cols: int, rng: np.random.Generator, latent_dim: int = 3) ->  pd.DataFrame:
    """
    Build a mutually correlated but globally irrelevant block of features generated via a synthetic join identifier.
    """
    latent = rng.normal(size=(n_rows, latent_dim))

    spurious_data: dict[str, np.ndarray | pd.Categorical] = {}

    n_numeric = n_cols // 2
    n_categorical = n_cols - n_numeric

    for j in range(n_numeric):
        weights = rng.normal(size=latent_dim)
        base_signal = latent @ weights
        noise = rng.normal(scale=0.15 * np.std(base_signal), size=n_rows)
        vals = base_signal + noise
        spurious_data[f"spurious_num_{j+1}"] = vals

    for j in range(n_categorical):
        weights = rng.normal(size=latent_dim)
        score = latent @ weights + rng.normal(scale=0.2, size=n_rows)

        q = np.quantile(score, [0.25, 0.5, 0.75])
        cats = np.where(score <= q[0], "A", np.where(score <= q[1], "B", np.where(score <= q[2], "C", "D")))

        spurious_data[f"spurious_cat_{j+1}"] = pd.Categorical(cats)

    spurious_df = pd.DataFrame(spurious_data)

    return spurious_df