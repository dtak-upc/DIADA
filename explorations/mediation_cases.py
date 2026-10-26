"""
A close look at single triangles A-B-C where V2 (explorations/lima_mediation.py) removed
A-C as "fully explained by B", to see what the data says and what LIMA's test sees.

One figure per case, four panels:
  (a) how A and C are related overall: P(C | A) on the bucketed data (each row sums to 1)
  (b) does knowing B remove it? the information A and C share (bits) overall, and within
      each value of B (bar width = share of rows with that B value); the weighted average
      of the bars is I(A;C | B)
  (c) what V2's test looks at: among row pairs that agree on B, how often they also agree
      on C, overall and split by whether they agree on A, with the z of each comparison
  (d) the V2 evidence (largest |z| over the four A/C patterns) against the number of row
      pairs sampled, 5 repetitions each; V2 used 2^17 pairs and removes the edge below 4

Pairs are drawn as LIMA's TPDomain does (ordered, with replacement, (i, i) -> (i, i+1)),
and the statistic is LIMA's own nodeDist4, as in lima_mediation.conditional_evidence.

Run with:  python explorations/mediation_cases.py
Writes results/jar_vs_two_sided/cases/<case>.png and cases.csv
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
from diada.core.diada_tool import _label_columns  # noqa: E402
from diada.core.LIMA.util.stats.betaDist import nodeDist4  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lima_mediation import _dist, conditional_evidence  # noqa: E402
from two_sided_plots import BANK  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided" / "cases"
SYN = paths.RESULTS_DIR / "synthetic_triads" / "data"
BASE = paths.DATA_DIR / "base_datasets"
# (name, csv, A, B (the column V2 said explains A-C), C, what it is)
CASES = [
    ("1_chain_A-C_via_B", SYN / "chain.csv", "A", "B", "C", "synthetic chain: true mediation (reference)"),
    ("2_house_bathrooms-price_via_sqft_living", BASE / "house_sales.csv", "bathrooms", "sqft_living", "price",
     "house_sales: a removal that makes sense"),
    ("3_bank_contact-Class_via_month", BASE / "bank.csv", "V9", "V11", "Class", "bank: removed, but month explains almost nothing"),
    ("4_bank_pdays-month_via_housing_loan", BASE / "bank.csv", "V14", "V7", "V11", "bank: removed, but housing_loan explains almost nothing"),
    ("5_house_price-yr_built_via_sqft_living", BASE / "house_sales.csv", "price", "sqft_living", "yr_built",
     "house_sales: removed, yet the link is stronger once sqft_living is known"),
]
SIZES = [1 << k for k in range(13, 22)]
V2_PAIRS = 1 << 17


def name(col: str) -> str:
    return BANK.get(col, col)


def bucketed(csv: Path, cols) -> pd.DataFrame:
    df = pd.read_csv(csv)
    lab = _label_columns(df.copy(), 10)
    lab.columns = df.columns
    return lab[list(cols)]


def ordered_codes(s: pd.Series):
    """Integer codes, ordered by value for buckets (intervals / numbers), by frequency otherwise."""
    vals = s.unique()
    try:
        order = sorted(vals, key=lambda v: v.left if isinstance(v, pd.Interval) else float(v))
    except (TypeError, ValueError):
        order = list(s.value_counts().index)
    index = {v: i for i, v in enumerate(order)}
    short = [f"{v.left:.4g}-{v.right:.4g}" if isinstance(v, pd.Interval) else str(v) for v in order]
    return s.map(index).to_numpy(), short


def mi(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information in bits, Miller-Madow corrected (as in the other scripts)."""
    def h(*cs):
        j = cs[0].astype(np.int64)
        for c in cs[1:]:
            j = j * (int(c.max()) + 1) + c
        cnt = np.unique(j, return_counts=True)[1]
        p = cnt / cnt.sum()
        return (-(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * cnt.sum())) / np.log(2)
    return h(x) + h(y) - h(x, y)


def sample_agree(codes: dict, n_pairs: int, rng: np.random.Generator) -> dict:
    n = len(next(iter(codes.values())))
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    j = np.where(i == j, (j + 1) % n, j)
    return {k: v[i] == v[j] for k, v in codes.items()}


def pattern_z(agree: dict, pa: bool, pc: bool) -> float:
    """z of 'C pc among pairs with A pa and B=' vs 'C pc among pairs with B=' (nodeDist4)."""
    n = len(agree["B"])
    a_side = agree["A"] if pa else ~agree["A"]
    c_side = agree["C"] if pc else ~agree["C"]
    on_b = agree["B"]
    return nodeDist4(_dist((on_b & a_side).sum(), n), _dist((on_b & a_side & c_side).sum(), n),
                     _dist(on_b.sum(), n), _dist((on_b & c_side).sum(), n))


def exact_agree_rates(ca, cb, cc) -> dict:
    """Share of ALL row pairs (i != j) with each agreement pattern, from the full data:
    agreeing on a set of columns = having the same value combination, so the count is
    sum over combinations of n * (n - 1)."""
    N = len(ca)
    def p(*cs):
        j = cs[0].astype(np.int64)
        for c in cs[1:]:
            j = j * (int(c.max()) + 1) + c
        n = np.unique(j, return_counts=True)[1].astype(float)
        return float((n * (n - 1)).sum() / (N * (N - 1)))
    return dict(B=p(cb), AB=p(ca, cb), BC=p(cb, cc), ABC=p(ca, cb, cc))


def expected_evidence(r: dict, n: int) -> float:
    """V2's evidence if a sample of n pairs had exactly the full data's pattern shares."""
    best = 0.0
    for pa in (True, False):
        for pc in (True, False):
            ab = r["AB"] if pa else r["B"] - r["AB"]
            bc = r["BC"] if pc else r["B"] - r["BC"]
            abc = {(True, True): r["ABC"], (True, False): r["AB"] - r["ABC"],
                   (False, True): r["BC"] - r["ABC"], (False, False): r["B"] - r["AB"] - r["BC"] + r["ABC"]}[(pa, pc)]
            z = nodeDist4(_dist(round(ab * n), n), _dist(round(abc * n), n), _dist(round(r["B"] * n), n), _dist(round(bc * n), n))
            best = max(best, abs(z))
    return best


def figure(case, rng) -> dict:
    key, csv, a, b, c, title = case
    df = bucketed(csv, (a, b, c))
    (ca, la), (cb, lb), (cc, lc) = (ordered_codes(df[x]) for x in (a, b, c))
    A, B, C = name(a), name(b), name(c)

    fig, ax = plt.subplots(2, 2, figsize=(17, 12.5))
    ax = ax.flat
    fig.suptitle(f"{title}\nEdge {A} - {C}, removed by V2 as explained by {B}", fontsize=14)

    # (a) P(C | A)
    tab = pd.crosstab(ca, cc).to_numpy().astype(float)
    cond = tab / tab.sum(1, keepdims=True)
    im = ax[0].imshow(cond, cmap="Blues", aspect="auto", vmin=0)
    ax[0].set_yticks(range(len(la)), [f"{l} (n={int(t)})" for l, t in zip(la, tab.sum(1))], fontsize=7)
    ax[0].set_xticks(range(len(lc)), lc, fontsize=7, rotation=60, ha="right")
    ax[0].set_ylabel(A), ax[0].set_xlabel(C)
    ax[0].set_title(f"(a) P({C} | {A}): rows differ = related", fontsize=11)
    fig.colorbar(im, ax=ax[0], fraction=0.04)

    # (b) I(A;C) overall and within each value of B
    overall = mi(ca, cc)
    within, weight = [], []
    for v in range(len(lb)):
        m = cb == v
        within.append(mi(ca[m], cc[m]) if m.sum() > 1 and len(np.unique(ca[m])) > 1 and len(np.unique(cc[m])) > 1 else 0.0)
        weight.append(m.mean())
    cmi = float(np.dot(within, weight))
    lefts = np.concatenate([[0], np.cumsum(weight)[:-1]])
    ax[1].bar(lefts, within, width=np.array(weight) * 0.95, align="edge", color="#378ADD", alpha=0.75,
              label=f"I({A};{C}) within each value of {B}")
    ax[1].axhline(overall, color="#E24B4A", linestyle="--", label=f"I({A};{C}) overall = {overall:.3f} bits")
    ax[1].axhline(cmi, color="#1D1D1B", label=f"I({A};{C} | {B}) = {cmi:.3f} bits ({cmi / overall:.0%} of overall)")
    ax[1].set_xticks(lefts + np.array(weight) / 2, lb, fontsize=7, rotation=60, ha="right")
    ax[1].set_xlabel(f"{B} (bar width = share of rows)")
    ax[1].set_ylabel("bits")
    ax[1].set_title(f"(b) is the {A}-{C} link still there once {B} is known?", fontsize=11)
    ax[1].legend(fontsize=8, loc="upper right")

    # (c) the pair-space view at V2's sample size
    ag = sample_agree({"A": ca, "B": cb, "C": cc}, V2_PAIRS, rng)
    on_b = ag["B"]
    p_c = ag["C"][on_b].mean()
    p_c_a = ag["C"][on_b & ag["A"]].mean()
    p_c_na = ag["C"][on_b & ~ag["A"]].mean()
    zs = {(pa, pc): pattern_z(ag, pa, pc) for pa in (True, False) for pc in (True, False)}
    bars = ax[2].bar([0, 1, 2], [p_c, p_c_a, p_c_na], color=["#B4B2A9", "#378ADD", "#D85A30"])
    ax[2].set_xticks([0, 1, 2], [f"all pairs\nwith {B}=\n(n={on_b.sum():,})",
                                 f"... and {A}=\n(n={(on_b & ag['A']).sum():,})",
                                 f"... and {A}≠\n(n={(on_b & ~ag['A']).sum():,})"], fontsize=8)
    for x, p, z in [(1, p_c_a, zs[(True, True)]), (2, p_c_na, zs[(False, True)])]:
        ax[2].text(x, p, f"{p:.3f}\nz = {z:+.2f}", ha="center", va="bottom", fontsize=9)
    ax[2].text(0, p_c, f"{p_c:.3f}", ha="center", va="bottom", fontsize=9)
    ax[2].set_ylim(0, min(1.15, max(p_c, p_c_a, p_c_na) * 1.6))
    ax[2].set_ylabel(f"share of pairs that also agree on {C}")
    best = max(abs(z) for z in zs.values())
    ax[2].set_title(f"(c) what V2's test sees (one sample of 2^17 pairs):\namong pairs with {B}=, the share that also have {C}=",
                    fontsize=11)
    ax[2].text(0.02, 0.97, "z of all four patterns:\n" + "\n".join(f"  {A}{'=' if pa else '≠'}, {C}{'=' if pc else '≠'}:  {z:+.2f}"
                                                              for (pa, pc), z in zs.items()) + f"\nevidence = largest |z| = {best:.2f}",
               transform=ax[2].transAxes, va="top", fontsize=9, family="monospace",
               bbox=dict(facecolor="white", edgecolor="#B4B2A9"))

    # (d) evidence vs sample size
    ev = np.array([[conditional_evidence(sample_agree({"A": ca, "B": cb, "C": cc}, n, rng), "A", "B", "C", True)
                    for _ in range(5)] for n in SIZES])
    for k in range(5):
        ax[3].scatter(SIZES, ev[:, k], color="#378ADD", s=14, alpha=0.6)
    ax[3].plot(SIZES, ev.mean(1), color="#378ADD", label="evidence (5 samples each, mean line)")
    exact = exact_agree_rates(ca, cb, cc)
    grid = [1 << k for k in range(13, 27)]
    ax[3].plot(grid, [expected_evidence(exact, n) for n in grid], color="#1D1D1B", linestyle="-.",
               label="expected, if the sample matched the full data exactly")
    ax[3].axhline(4, color="#E24B4A", linestyle="--", label="V2 removes the edge below 4")
    ax[3].axvline(V2_PAIRS, color="#888780", linestyle=":", label="pairs V2 used (2^17)")
    ax[3].set_xscale("log", base=2)
    ax[3].set_xlabel("row pairs sampled")
    ax[3].set_ylabel("largest |z|")
    ax[3].set_title("(d) does more data change the verdict?", fontsize=11)
    ax[3].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / f"{key}.png", dpi=75)
    plt.close(fig)
    return dict(case=key, A=A, B=B, C=C, mi_bits=round(overall, 4), mi_given_B=round(cmi, 4), share_left=round(cmi / overall, 2),
                pairs_agreeing_on_B=int(on_b.sum()), p_C_agree=round(p_c, 4), p_C_agree_if_A_agree=round(p_c_a, 4),
                p_C_agree_if_A_differ=round(p_c_na, 4), evidence_2_17=round(best, 2),
                exact_P_C_given_B=round(exact["BC"] / exact["B"], 4), exact_P_C_given_AB=round(exact["ABC"] / exact["AB"], 4),
                pairs_needed_for_4=next((f"2^{k}" for k in range(13, 34) if expected_evidence(exact, 1 << k) > 4), ">2^33"),
                **{f"evidence_2^{int(np.log2(n))}": round(float(e), 2) for n, e in zip(SIZES, ev.mean(1))})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for case in CASES:
        rows.append(figure(case, rng))
        print(f"{case[0]} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cases.csv", index=False)
    print(t.T.to_string())


if __name__ == "__main__":
    main()
