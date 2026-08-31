import numpy as np
import pandas as pd

def random_numeric_noise(n: int, variant: int, rng: np.random.Generator) -> np.ndarray:
    choice = variant % 7
    if choice == 0:
        return rng.normal(0, 1, n)
    elif choice == 1:
        return rng.exponential(scale=2.0, size=n)
    elif choice == 2:
        return rng.uniform(-10, 10, size=n)
    elif choice == 3:
        return rng.standard_t(df=3, size=n)
    elif choice == 4: # Bimodal
        mask = rng.random(n) < 0.5
        out = np.where(mask, rng.normal(-3, 1, n), rng.normal(3, 1, n))
        return out
    elif choice == 5:
        return rng.lognormal(mean=0, sigma=1, size=n)
    else: # Integer-like Poisson
        return rng.poisson(lam=5, size=n).astype(float)


def random_categorical_noise(n: int, variant: int, rng: np.random.Generator) -> pd.Categorical:
    choice = variant % 4
    if choice == 0:
        categories = list("ABCDE")
        probs = [0.4, 0.3, 0.15, 0.1, 0.05]
    elif choice == 1:
        categories = [f"cat_{k}" for k in range(10)]
        probs = np.ones(10) / 10
    elif choice == 2:
        categories = ["yes", "no"]
        probs = [0.7, 0.3]
    else:
        categories = [f"group_{k}" for k in range(4)]
        probs = [0.5, 0.25, 0.15, 0.10]

    probs = np.array(probs)
    probs = probs / probs.sum()
    return pd.Categorical(rng.choice(categories, size=n, p=probs))


def shuffle_column(values: np.ndarray, corruption_rate: float, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    result = values.copy()

    n_corrupt = int(round(corruption_rate * n))
    positions = rng.choice(n, size=n_corrupt, replace=False) # Positions to corrupt
    replacements = rng.choice(values, size=n_corrupt, replace=True) # Draw replacement values from the pool of original values
    result[positions] = replacements

    return result