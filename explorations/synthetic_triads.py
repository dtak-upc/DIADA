"""
Three-variable synthetic benchmarks, one per kind of relationship between
columns A, B and C.

Every column is discrete with few values (prefixed strings such as "a0", "b3"),
so DIADA's bucketing leaves them unchanged: every value is a category, and
every number can be checked by hand.

    independent      A, B, C unrelated
    pair_isolated    B = noisy copy of A; C unrelated to both
    chain            A -> B -> C (noisy copies)
    common_cause     B -> A and B -> C (noisy copies of B)
    hub_join         two tables, one with A per key and one with C per key, joined
                     on a 20-value key B (every combination of rows within a key)
    direct_triangle  every pair shares its own hidden bit, unknown to the third column:
                     A = (bit_AB, bit_AC), B = (bit_AB, bit_BC), C = (bit_AC, bit_BC),
                     each column holding its own copy of its bits with 5% of them flipped
    sum_collider     A, C unrelated; B = A + C
    xor              A, C unrelated bits; B = A xor C
    interaction      C moderates how A depends on B: B = (S, U), C = (S', W) with S' a noisy
                     copy of S (10% flipped), and A = U when W = 0, a random bit when W = 1.
                     A-B and B-C are related, A-C are not, and knowing C on top of B tells
                     more about A than B alone. Like sum_collider, but here A is the outcome
                     (of B, moderated by C) instead of B being the common outcome of A and C.

Expected results come in two layers (see EXPECTED below):
    graph     which pairs are related on their own (overall); these are the edges a
              correct pairwise graph should contain
    decision  whether a connected A-B-C should stay one group ("don't split") or be cut
              at B into {A, B} and {B, C} ("split": A and C share nothing once B is known)
"Don't split" does not mean A-C should be an edge: in sum_collider A and C are only
related together with B, so the graph is A-B-C without A-C, and the group stays whole.

The script also checks each benchmark: for every pair it prints how much
information the two columns share (mutual information, in bits) overall and
once the third column is known (conditional mutual information).

Run with:  python explorations/synthetic_triads.py
Writes results/synthetic_triads/data/<name>.csv
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from diada import paths

OUT_DIR = paths.RESULTS_DIR / "synthetic_triads" / "data"
N = 5000
LEVELS = 4
TRIANGLE_NOISE = 0.05  # share of each hidden bit flipped in each column's copy (direct_triangle)
SHARED_BIT_NOISE = 0.10  # share of C's copy of the bit it shares with B that is flipped (interaction)

# benchmark -> (graph: pairs related on their own, decision for a connected A-B-C)
EXPECTED = {
    "independent": ("none", "three separate columns"),
    "pair_isolated": ("A-B", "{A, B}; C separate"),
    "chain": ("A-B, B-C, A-C", "split: {A, B}, {B, C}"),
    "common_cause": ("A-B, B-C, A-C", "split: {A, B}, {B, C}"),
    "hub_join": ("A-B, B-C, weak A-C", "split: {A, B}, {B, C}"),
    "direct_triangle": ("A-B, B-C, A-C", "don't split: {A, B, C}"),
    "sum_collider": ("A-B, B-C (no A-C)", "don't split: {A, B, C}"),
    "xor": ("none", "only a three-way relationship; invisible to pairwise methods"),
    "interaction": ("A-B, B-C (no A-C)", "don't split: {A, B, C}"),
}

# The graph layer in machine-readable form: benchmark -> (edges that must appear, edges that may).
# hub_join's A-C is a real but weak overall link (0.07 bits), so finding it or not are both fine.
EXPECTED_EDGES = {
    "independent": (set(), set()),
    "pair_isolated": ({"A-B"}, set()),
    "chain": ({"A-B", "B-C", "A-C"}, set()),
    "common_cause": ({"A-B", "B-C", "A-C"}, set()),
    "hub_join": ({"A-B", "B-C"}, {"A-C"}),
    "direct_triangle": ({"A-B", "B-C", "A-C"}, set()),
    "sum_collider": ({"A-B", "B-C"}, set()),
    "xor": (set(), set()),
    "interaction": ({"A-B", "B-C"}, set()),
}


def noisy_copy(x: np.ndarray, keep: float, rng: np.random.Generator, levels: int = LEVELS) -> np.ndarray:
    """x with probability `keep`, otherwise a uniformly random value."""
    return np.where(rng.random(len(x)) < keep, x, rng.integers(0, levels, len(x)))


def hidden_bit_triangle(rng: np.random.Generator, noise: float = TRIANGLE_NOISE) -> dict:
    """Each pair of columns shares its own hidden bit that the third column knows nothing
    about, so every pair is related directly and none through the third column. Each
    column stores its two bits as one 4-valued value, with each bit flipped with
    probability `noise` (independently per column)."""
    ab, ac, bc = (rng.integers(0, 2, N) for _ in range(3))
    flip = lambda bits: np.where(rng.random(N) < noise, 1 - bits, bits)  # noqa: E731
    return dict(A=2 * flip(ab) + flip(ac), B=2 * flip(ab) + flip(bc), C=2 * flip(ac) + flip(bc))


def moderated_interaction(rng: np.random.Generator, noise: float = SHARED_BIT_NOISE) -> dict:
    """C moderates the dependence of A on B. B = (S, U) and C = (S', W) share the bit S
    (C's copy flipped with probability `noise`), so B and C are related. A copies U when
    W = 0 and is a random bit when W = 1: A depends on B, C alone says nothing about A's
    value, but knowing C on top of B tells whether A follows U."""
    s, u, w = (rng.integers(0, 2, N) for _ in range(3))
    s_copy = np.where(rng.random(N) < noise, 1 - s, s)
    a = np.where(w == 0, u, rng.integers(0, 2, N))
    return dict(A=a, B=2 * s + u, C=2 * s_copy + w)


def make_benchmarks(rng: np.random.Generator) -> dict:
    u = lambda: rng.integers(0, LEVELS, N)  # noqa: E731
    b = {}

    b["independent"] = dict(A=u(), B=u(), C=u())

    a = u()
    b["pair_isolated"] = dict(A=a, B=noisy_copy(a, 0.7, rng), C=u())

    a = u()
    mid = noisy_copy(a, 0.7, rng)
    b["chain"] = dict(A=a, B=mid, C=noisy_copy(mid, 0.7, rng))

    root = u()
    b["common_cause"] = dict(A=noisy_copy(root, 0.7, rng), B=root, C=noisy_copy(root, 0.7, rng))

    # Data-lake hub: each key has its own distribution over A (table 1) and over C (table 2);
    # the join pairs every table-1 row of a key with every table-2 row of that key.
    keys, rows_per_key = 20, 16
    pa = rng.dirichlet(np.full(LEVELS, 0.5), size=keys)
    pc = rng.dirichlet(np.full(LEVELS, 0.5), size=keys)
    hub = {"A": [], "B": [], "C": []}
    for k in range(keys):
        a_rows = rng.choice(LEVELS, size=rows_per_key, p=pa[k])
        c_rows = rng.choice(LEVELS, size=rows_per_key, p=pc[k])
        for av, cv in itertools.product(a_rows, c_rows):
            hub["A"].append(av), hub["B"].append(k), hub["C"].append(cv)
    b["hub_join"] = {k: np.array(v) for k, v in hub.items()}

    b["direct_triangle"] = hidden_bit_triangle(np.random.default_rng(1))
    # The previous direct_triangle drew from this stream here; take the same draws so that
    # sum_collider and xor below keep exactly the same data as before.
    rng.integers(0, LEVELS, N), rng.random(N), rng.integers(0, LEVELS, N), rng.random(N), rng.integers(0, LEVELS, N)

    a, c = u(), u()
    b["sum_collider"] = dict(A=a, B=a + c, C=c)

    a, c = rng.integers(0, 2, N), rng.integers(0, 2, N)
    b["xor"] = dict(A=a, B=a ^ c, C=c)

    b["interaction"] = moderated_interaction(np.random.default_rng(2))

    return {name: pd.DataFrame({col: np.array([f"{col.lower()}{v}" for v in vals]) for col, vals in cols.items()})
            for name, cols in b.items()}


def entropy(*cols: np.ndarray) -> float:
    """Joint entropy in bits, with the Miller-Madow small-sample correction."""
    joint = pd.Series(list(zip(*cols))) if len(cols) > 1 else pd.Series(cols[0])
    counts = joint.value_counts().to_numpy()
    p = counts / counts.sum()
    return (-(p * np.log(p)).sum() + (len(counts) - 1) / (2 * counts.sum())) / np.log(2)


def check(name: str, df: pd.DataFrame) -> list:
    rows = []
    for x, z in itertools.combinations("ABC", 2):
        y = next(c for c in "ABC" if c not in (x, z))
        X, Y, Z = df[x].to_numpy(), df[y].to_numpy(), df[z].to_numpy()
        mi = entropy(X) + entropy(Z) - entropy(X, Z)
        cmi = entropy(X, Y) + entropy(Z, Y) - entropy(X, Y, Z) - entropy(Y)
        rows.append(dict(benchmark=name, pair=f"{x}-{z}", given=y, I=round(mi, 3), I_given=round(cmi, 3)))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, df in make_benchmarks(np.random.default_rng(0)).items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows, values per column: " + ", ".join(f"{c}={df[c].nunique()}" for c in df.columns))
        rows += check(name, df)
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(pd.DataFrame([dict(benchmark=k, graph=g, decision=d) for k, (g, d) in EXPECTED.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
