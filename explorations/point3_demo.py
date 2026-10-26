"""
Point 3 of the methodology review: combining statements "A | S u {X} = A | S" (X adds nothing about A
once S is known) into minimal sets of columns.

  (a) Two columns that are each redundant given the other, but not together:
      - near-duplicates (diamonds: y and z are two measurements of the same size; synthetic: two
        98% copies of the same hidden value): each is redundant given the other, but removing both
        loses what they say about the target
      - no duplicates (synthetic: A depends on C; B and D are noisy readings of C): both can go
  (b) A redundancy does not carry over to a bigger set: sum_collider (B = A + C): C says nothing about
      A on its own, but once B is known C tells A exactly
Measured with the conditional mutual information on the bucketed values (bits), the reference
measure.

Run with:  python explorations/point3_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diada import paths

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402


def H(*cols) -> float:
    if not cols:
        return 0.0
    j = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        j = j * (int(c.max()) + 1) + c
    _, cnt = np.unique(j, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum() + (len(cnt) - 1) / (2 * cnt.sum() * np.log(2)))


def I(a, xs, given=()) -> float:
    """I(a; xs | given) in bits (Miller-Madow); xs and given are lists of columns."""
    xs, given = list(xs), list(given)
    return H(a, *given) + H(*xs, *given) - H(a, *xs, *given) - H(*given)


def main() -> None:
    d = bucket_codes(pd.read_csv(paths.DATA_DIR / "base_datasets" / "diamonds.csv"))
    price, y, z, cut = d["price"], d["y"], d["z"], d["cut"]
    both = I(price, [y, z], [cut])
    print("(a1) diamonds, target price, context cut; y (width) and z (depth) are two measurements of size")
    print(f"   y and z together, given cut:  I(price; y,z | cut) = {both:.3f} bits")
    print(f"   y given z and cut:            I(price; y | z,cut) = {I(price, [y], [z, cut]):.3f}  ({I(price, [y], [z, cut]) / both:.0%} of it)")
    print(f"   z given y and cut:            I(price; z | y,cut) = {I(price, [z], [y, cut]):.3f}  ({I(price, [z], [y, cut]) / both:.0%} of it)")
    print("   -> each looks removable given the other, but removing both loses everything they say")

    rng = np.random.default_rng(0)
    n = 50000
    latent = rng.integers(0, 4, n)
    noisy = lambda v, keep: np.where(rng.random(n) < keep, v, rng.integers(0, 4, n))  # noqa: E731
    a = noisy(latent, 0.6)
    b, dd = noisy(latent, 0.98), noisy(latent, 0.98)
    c = rng.integers(0, 3, n)
    print("(a2) synthetic duplicates: B and D are 98% copies of the same hidden value, A depends on it, C is unrelated")
    print(f"   I(A; B | C,D) = {I(a, [b], [c, dd]):.4f} | I(A; D | B,C) = {I(a, [dd], [b, c]):.4f} | I(A; B,D | C) = {I(a, [b, dd], [c]):.4f}")
    c2 = rng.integers(0, 4, n)
    a2 = noisy(c2, 0.6)
    b2, d2 = noisy(c2, 0.6), noisy(c2, 0.6)
    print("(a3) synthetic, no duplicates: A depends on C; B and D are separate noisy readings of C")
    print(f"   I(A; B | C,D) = {I(a2, [b2], [c2, d2]):.4f} | I(A; D | B,C) = {I(a2, [d2], [b2, c2]):.4f} | "
          f"I(A; B,D | C) = {I(a2, [b2, d2], [c2]):.4f}  -> both can be removed together")

    s = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / "sum_collider.csv")
    A, B, C = (pd.factorize(s[k])[0] for k in "ABC")
    print("(b) sum_collider: B = A + C")
    print(f"   C about A on its own:      I(A; C) = {I(A, [C]):.3f} bits   -> C looks redundant for A")
    print(f"   C about A once B is known: I(A; C | B) = {I(A, [C], [B]):.3f} bits -> but with B in the set, C is needed")
    print(f"   B alone about A: I(A; B) = {I(A, [B]):.3f} | B and C together: I(A; B,C) = {I(A, [B, C]):.3f} (A has H = {H(A):.3f})")


if __name__ == "__main__":
    main()
