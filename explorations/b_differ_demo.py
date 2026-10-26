"""
Why "A and C are related beyond B" must be tested on pairs with the same value of B (B=), and why
pairs that differ on B (B!=) give false evidence, while either kind of pair is fine for "A and C
are related at all".

Statistic (LIMA's predicates, exact counts over all row pairs): phi = the correlation between
"the pair agrees on A" and "the pair agrees on C", within a group of pairs:
    all pairs           what V1 measures: A and C related by any route
    B!= pairs           pairs whose two rows have different values of B
    B= pairs            pairs whose two rows have the same value v of B, phi per v, combined as
                        the average weighted by each value's share of those pairs (signed, so
                        chance does not add up)

Control: shuffle A within each value of B. This keeps A-B and B-C exactly as they are and
destroys any A-C link beyond B. A statistic that still shows a link after the shuffle is showing
B's effect, not an A-C link.

Datasets: A and C unrelated once B is known (hub with a 2-value key; hub with 20 keys, the
synthetic hub_join; a chain through an ORDERED B, e.g. a bucketed number), a direct triangle
(A and C really related beyond B) as contrast, and two real triangles (house_sales bathrooms-price
through sqft_living, an ordered B; bank previous-Class through poutcome, an unordered B).

Run with:  python explorations/b_differ_demo.py
Writes results/b_differ/{summary.csv, distance.csv, b_differ.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from exact_counts import bucket_codes  # noqa: E402

OUT = paths.RESULTS_DIR / "b_differ"
N = 20000
SHUFFLES = 20


def same(*cols) -> float:
    """Ordered pairs (i != j) agreeing on all cols."""
    j = cols[0].astype(np.int64)
    for c in cols[1:]:
        j = j * (int(c.max()) + 1) + c
    n = np.bincount(j).astype(float)
    return float((n * (n - 1)).sum())


def phi(t, a, c, ac) -> float:
    pa, pc, pac = a / t, c / t, ac / t
    den = np.sqrt(pa * (1 - pa) * pc * (1 - pc))
    return float((pac - pa * pc) / den) if den > 0 else float("nan")


def statistics(a, b, c) -> dict:
    """phi over all pairs, over B!= pairs, and per value of B (combined)."""
    n = len(a)
    t_all, a_all, c_all, ac_all = n * (n - 1.0), same(a), same(c), same(a, c)
    t_eq, a_eq, c_eq, ac_eq = same(b), same(b, a), same(b, c), same(b, a, c)
    per_value, weights = [], []
    for v in np.unique(b):
        m = b == v
        k = m.sum()
        if k < 2:
            continue
        f = phi(k * (k - 1.0), same(a[m]), same(c[m]), same(a[m], c[m]))
        if f == f:
            per_value.append(f)
            weights.append(k * (k - 1.0))
    return dict(all_pairs=phi(t_all, a_all, c_all, ac_all),
                b_differ=phi(t_all - t_eq, a_all - a_eq, c_all - c_eq, ac_all - ac_eq),
                b_same=float(np.average(per_value, weights=weights)))


def shuffle_within(a, b, rng):
    out = a.copy()
    for v in np.unique(b):
        idx = np.flatnonzero(b == v)
        out[idx] = a[rng.permutation(idx)]
    return out


def by_distance(a, b, c) -> pd.DataFrame:
    """For an ordered B: among pairs whose B values are d buckets apart, how often they agree on
    A, on C, on both, and phi within that group."""
    nb = int(b.max()) + 1
    na, nc = int(a.max()) + 1, int(c.max()) + 1
    n_b = np.bincount(b, minlength=nb).astype(float)
    n_ba = np.bincount(b * na + a, minlength=nb * na).reshape(nb, na).astype(float)
    n_bc = np.bincount(b * nc + c, minlength=nb * nc).reshape(nb, nc).astype(float)
    n_bac = np.bincount((b * na + a) * nc + c, minlength=nb * na * nc).reshape(nb, na * nc).astype(float)
    T, SA, SC, SAC = np.outer(n_b, n_b), n_ba @ n_ba.T, n_bc @ n_bc.T, n_bac @ n_bac.T
    rows = []
    for d in range(1, nb):
        mask = np.abs(np.subtract.outer(np.arange(nb), np.arange(nb))) == d
        t, sa, sc, sac = T[mask].sum(), SA[mask].sum(), SC[mask].sum(), SAC[mask].sum()
        rows.append(dict(distance=d, pairs=t, agree_A=sa / t, agree_C=sc / t, agree_both=sac / t, phi_within=phi(t, sa, sc, sac)))
    return pd.DataFrame(rows)


def datasets(rng):
    out = {}
    # hub with a 2-value key: A and C each depend on the key, nothing else
    b = rng.integers(0, 2, N)
    noisy = lambda base: np.where(rng.random(N) < 0.7, base, rng.integers(0, 4, N))  # noqa: E731
    out["hub, 2-value key"] = (noisy(np.where(b == 0, 0, 3)), b, noisy(np.where(b == 0, 1, 2)))
    # hub_join: 20 keys (synthetic_triads)
    h = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / "hub_join.csv")
    out["hub, 20 keys (hub_join)"] = tuple(pd.factorize(h[x])[0] for x in "ABC")
    # chain through an ordered B: B is a number in 10 buckets, A and C are B plus independent noise
    x = rng.random(N) * 10
    bucket = lambda v: np.clip(np.floor(v), 0, 9).astype(np.int64)  # noqa: E731
    out["chain through ordered B"] = (bucket(x + rng.normal(0, 1.2, N)), bucket(x), bucket(x + rng.normal(0, 1.8, N)))
    # direct triangle: A and C share their own bit (synthetic_triads)
    t = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / "direct_triangle.csv")
    out["direct triangle (real A-C link)"] = tuple(pd.factorize(t[x])[0] for x in "ABC")
    hs = bucket_codes(pd.read_csv(paths.DATA_DIR / "base_datasets" / "house_sales.csv"))
    out["house_sales: bathrooms-price via sqft_living"] = (hs["bathrooms"], hs["sqft_living"], hs["price"])
    bk = bucket_codes(pd.read_csv(paths.DATA_DIR / "base_datasets" / "bank.csv"))
    out["bank: previous-Class via poutcome"] = (bk["V15"], bk["V16"], bk["Class"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for name, (a, b, c) in datasets(rng).items():
        obs = statistics(a, b, c)
        null = [statistics(shuffle_within(a, b, rng), b, c) for _ in range(SHUFFLES)]
        row = dict(dataset=name, values_of_B=len(np.unique(b)))
        for k in ("all_pairs", "b_differ", "b_same"):
            vals = np.array([s[k] for s in null])
            row[k] = round(obs[k], 4)
            row[k + "_after_shuffle"] = round(float(vals.mean()), 4)
            row[k + "_shuffle_sd"] = round(float(vals.std()), 4)
        rows.append(row)
        print(name, "done", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    a, b, c = datasets(np.random.default_rng(0))["chain through ordered B"]
    dist = by_distance(a, b, c)
    dist.to_csv(OUT / "distance.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(19, 7.2), gridspec_kw=dict(width_ratios=[1, 1.9]))
    same_b = (same(b, a) / same(b), same(b, c) / same(b))
    ax[0].plot(dist.distance, dist.agree_A, marker="o", color="#378ADD", label="share agreeing on A")
    ax[0].plot(dist.distance, dist.agree_C, marker="s", color="#D85A30", label="share agreeing on C")
    ax[0].scatter([0, 0], same_b, color=["#378ADD", "#D85A30"], s=60, zorder=3)
    ax[0].axvline(0.5, color="#B4B2A9", linestyle=":")
    ax[0].text(0.05, max(same_b) * 1.02, "B=", fontsize=10)
    ax[0].text(0.6, max(same_b) * 1.02, "B!= (by how far apart)", fontsize=10)
    ax[0].set_xlabel("distance between the two rows' B buckets")
    ax[0].set_ylabel("share of pairs")
    ax[0].set_title("(a) chain through an ordered B: among B!= pairs, rows whose B values\n"
                    "are close agree more on A AND on C; far apart, less on both", fontsize=11)
    ax[0].legend(fontsize=9)

    names = list(summary.dataset)
    xs = np.arange(len(names))
    w = 0.13
    groups = [("all_pairs", "all pairs", "#888780"), ("b_differ", "B!= pairs", "#E24B4A"), ("b_same", "B= pairs (per value)", "#378ADD")]
    for i, (k, lab, col) in enumerate(groups):
        ax[1].bar(xs + (2 * i - 2.5) * w, summary[k], width=w, color=col, label=f"{lab}: data")
        ax[1].bar(xs + (2 * i - 1.5) * w, summary[k + "_after_shuffle"], width=w, color=col, alpha=0.35, hatch="//",
                  label=f"{lab}: A shuffled within B (no A-C link beyond B)")
    ax[1].axhline(0, color="#5F5E5A", linewidth=0.8)
    short = {"hub, 2-value key": "hub,\n2-value key", "hub, 20 keys (hub_join)": "hub, 20 keys\n(hub_join)",
             "chain through ordered B": "chain through\nordered B", "direct triangle (real A-C link)": "direct triangle\n(real A-C link)",
             "house_sales: bathrooms-price via sqft_living": "house_sales:\nbathrooms-price\nvia sqft_living",
             "bank: previous-Class via poutcome": "bank:\nprevious-Class\nvia poutcome"}
    ax[1].set_xticks(xs, [short.get(n, n) for n in names], fontsize=9)
    ax[1].set_ylabel("phi: correlation of 'agree on A' and 'agree on C'")
    ax[1].set_title("(b) which group of pairs still shows an A-C link once any link beyond B is destroyed?", fontsize=11)
    ax[1].legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "b_differ.png", dpi=85)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(dist.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
