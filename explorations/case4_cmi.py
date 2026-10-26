"""
Case 4 of mediation_cases.py (bank, pdays - month given housing_loan) measured on values
instead of row pairs: the conditional mutual information I(pdays; month | housing_loan),
on the bucketed data DIADA uses.

  (a) where it comes from: the contribution of each month, within each housing_loan value
  (b) is it more than chance? the observed value against 500 shuffles of pdays within each
      housing_loan value (which keeps pdays-housing_loan and month-housing_loan as they are
      and breaks only pdays-month given housing_loan), plus the G-test approximation
  (c) how it behaves with the number of rows: random subsets of rows, the estimate against
      the shuffled 95th percentile, for case 4 and for the synthetic chain (A-C given B,
      true mediation) as a reference

Run with:  python explorations/case4_cmi.py
Writes results/jar_vs_two_sided/cases/4_detail_cmi.png and 4_detail_cmi.csv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from diada import paths  # noqa: E402
from diada.core.diada_tool import _label_columns  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided" / "cases"
N_SHUFFLES = 500


def codes(s: pd.Series):
    c, labels = pd.factorize(s.astype(str))
    return c, list(labels)


def table(a, b, c, na, nb, nc) -> np.ndarray:
    """Counts n[b, a, c]."""
    t = np.zeros(nb * na * nc)
    np.add.at(t, (b * na + a) * nc + c, 1)
    return t.reshape(nb, na, nc)


def cmi(t: np.ndarray, miller_madow: bool = False) -> float:
    """I(A; C | B) in bits from counts n[b, a, c]; optionally Miller-Madow corrected."""
    n = t.sum()

    def h(counts):
        counts = counts[counts > 0]
        p = counts / n
        return -(p * np.log2(p)).sum() + (miller_madow * (len(counts) - 1) / (2 * n * np.log(2)))

    return h(t.sum(2).ravel()) + h(t.sum(1).ravel()) - h(t.ravel()) - h(t.sum((1, 2)))


def shuffled(a, b, rng):
    """a shuffled within each value of b."""
    out = a.copy()
    for v in np.unique(b):
        idx = np.flatnonzero(b == v)
        out[idx] = a[rng.permutation(idx)]
    return out


def null_cmis(a, b, c, dims, rng, k=N_SHUFFLES):
    return np.array([cmi(table(shuffled(a, b, rng), b, c, *dims)) for _ in range(k)])


def main() -> None:
    rng = np.random.default_rng(0)
    df = pd.read_csv(paths.DATA_DIR / "base_datasets" / "bank.csv")
    lab = _label_columns(df.copy(), 10)
    lab.columns = df.columns
    (a, la), (b, lb), (c, lc) = codes(lab.V14.map(lambda v: "low" if v.left < 0 else "high")), codes(lab.V7), codes(lab.V11)
    dims = (len(la), len(lb), len(lc))
    t = table(a, b, c, *dims)
    N = len(a)

    # (a) contributions: p(b) * sum_a p(a,c|b) log p(a,c|b) / (p(a|b) p(c|b)), per month c and value b
    p_b = t.sum((1, 2)) / N
    contrib = np.zeros((dims[1], dims[2]))
    within = []
    for v in range(dims[1]):
        tb = t[v] / t[v].sum()
        pa, pc = tb.sum(1, keepdims=True), tb.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(tb > 0, tb * np.log2(tb / (pa * pc)), 0.0)
        contrib[v] = p_b[v] * terms.sum(0)
        within.append(terms.sum())
    mi_overall = cmi(t.sum(0)[None])
    observed, observed_mm = cmi(t), cmi(t, miller_madow=True)

    # (b) shuffles and G-test
    null = null_cmis(a, b, c, dims, rng)
    df_g = dims[1] * (dims[0] - 1) * (dims[2] - 1)
    g = 2 * N * np.log(2) * observed
    p_g = chi2.sf(g, df_g)

    # (c) behaviour with the number of rows, case 4 and the synthetic chain
    chain = pd.read_csv(paths.RESULTS_DIR / "synthetic_triads" / "data" / "chain.csv")
    (ca, _), (cb, _), (cc, _) = codes(chain.A), codes(chain.B), codes(chain.C)
    series = {"case 4: pdays-month | housing_loan": (a, b, c, dims, [250, 500, 1000, 2000, 5000, 10000, 20000, N]),
              "chain: A-C | B (true mediation)": (ca, cb, cc, (4, 4, 4), [250, 500, 1000, 2000, 5000])}
    rows = []
    for name, (x, y, z, d, sizes) in series.items():
        for n in sizes:
            for rep in range(5 if n < len(x) else 1):
                idx = rng.choice(len(x), n, replace=False) if n < len(x) else np.arange(len(x))
                tt = table(x[idx], y[idx], z[idx], *d)
                nul = null_cmis(x[idx], y[idx], z[idx], d, rng, k=100)
                obs = cmi(tt)
                rows.append(dict(series=name, rows=n, rep=rep, cmi=obs, cmi_miller_madow=cmi(tt, True),
                                 null_mean=nul.mean(), null_95=np.quantile(nul, 0.95), p_shuffle=(1 + (nul >= obs).sum()) / 101))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "4_detail_cmi.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1.1]))
    fig.suptitle("Case 4 on values: I(pdays; month | housing_loan), bucketed data", fontsize=14)
    order = np.argsort(-t.sum((0, 1)))
    x = np.arange(dims[2])
    for v, color in zip(range(dims[1]), ("#378ADD", "#D85A30")):
        ax[0].bar(x + (v - 0.5) * 0.4, contrib[v][order], width=0.4, color=color,
                  label=f"housing_loan = {lb[v]}: I within = {within[v]:.3f} bits, weight {p_b[v]:.2f}")
    ax[0].set_xticks(x, [lc[i] for i in order])
    ax[0].set_ylabel("bits contributed to I(pdays; month | housing_loan)")
    ax[0].set_title(f"(a) where it comes from: total {observed:.4f} bits (Miller-Madow {observed_mm:.4f}),\n"
                    f"vs. I(pdays; month) = {mi_overall:.4f} bits without conditioning", fontsize=11)
    ax[0].axhline(0, color="#888780", linewidth=0.8)
    ax[0].legend(fontsize=9)

    ax[1].hist(null, bins=30, color="#B4B2A9", label=f"{N_SHUFFLES} shuffles of pdays within housing_loan")
    ax[1].axvline(observed, color="#E24B4A", linewidth=2, label=f"observed {observed:.4f} bits")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("I(pdays; month | housing_loan), bits (log scale)")
    ax[1].set_title(f"(b) more than chance? shuffles: max {null.max():.5f}, observed is {observed / null.mean():.0f}x their mean\n"
                    f"G-test: G = {g:,.0f} on {df_g} degrees of freedom, p = {p_g:.1e}", fontsize=11)
    ax[1].legend(fontsize=9)

    for (name, s), color in zip(res.groupby("series", sort=False), ("#378ADD", "#888780")):
        ax[2].scatter(s.rows, s.cmi, color=color, s=16, alpha=0.7)
        m = s.groupby("rows").mean(numeric_only=True)
        ax[2].plot(m.index, m.cmi, color=color, label=f"{name}: estimate")
        ax[2].plot(m.index, m.null_95, color=color, linestyle="--", label=f"{name}: 95% of shuffles")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("rows used")
    ax[2].set_ylabel("bits")
    ax[2].set_title("(c) with fewer rows: estimate vs. what chance gives", fontsize=11)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "4_detail_cmi.png", dpi=80)
    plt.close(fig)

    print(f"I(pdays; month) = {mi_overall:.4f}; I(pdays; month | housing_loan) = {observed:.4f} (MM {observed_mm:.4f})")
    for v in range(dims[1]):
        print(f"  within housing_loan={lb[v]}: {within[v]:.4f} bits, weight {p_b[v]:.3f}; top months: "
              + ", ".join(f"{lc[i]} {contrib[v][i]:+.4f}" for i in np.argsort(-np.abs(contrib[v]))[:4]))
    print(f"shuffles: mean {null.mean():.5f}, max {null.max():.5f}; G = {g:.0f}, df = {df_g}, p = {p_g:.2e}")
    print(res.groupby(["series", "rows"]).mean(numeric_only=True)[["cmi", "cmi_miller_madow", "null_mean", "null_95", "p_shuffle"]].round(5).to_string())


if __name__ == "__main__":
    main()
