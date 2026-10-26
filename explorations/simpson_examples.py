"""
Two real cases of Simpson's paradox in bank, on DIADA's bucketed data:

  (a) In our own test (pairs of rows): pdays-month given housing_loan. Within the pairs where both
      rows have housing_loan = yes, and within those where both have no, rows that differ on pdays
      share a month LESS often. Pooled ("the two rows agree on housing_loan"), it reverses.
  (b) In the values: Class vs pdays given poutcome. Overall, high pdays (a long time since the last
      contact) goes with more "yes"; within every group of clients contacted before, it is the
      other way round. DIADA's bucketing puts "never contacted" (pdays = -1) in the same bucket as
      "contacted 0-185 days ago".

Run with:  python explorations/simpson_examples.py
Writes results/simpson/{simpson_examples.png, numbers.txt}
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402
from diada.core.diada_tool import _label_columns  # noqa: E402

OUT = paths.RESULTS_DIR / "simpson"
BLUE, ORANGE, GREY = "#378ADD", "#D85A30", "#888780"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(paths.DATA_DIR / "base_datasets" / "bank.csv")
    lab = _label_columns(df.copy(), 10)
    lab.columns = df.columns
    low = np.array([v.left < 0 for v in lab.V14])  # pdays bucket (-1, 185] vs (185, 871]
    housing, month, pout = df.V7.to_numpy(), df.V11.to_numpy(), df.V16.to_numpy()
    yes = (df.Class == 2).to_numpy()
    lines = []

    # (a) pairs of rows: share sharing a month, by housing_loan group and pdays agreement
    def pair_rates(mask):
        lo, hi = mask & low, mask & ~low
        m_lo, m_hi = pd.Series(month[lo]).value_counts(), pd.Series(month[hi]).value_counts()
        ll = (lo.sum() ** 2 - lo.sum(), (m_lo * (m_lo - 1)).sum())
        hh = (hi.sum() ** 2 - hi.sum(), (m_hi * (m_hi - 1)).sum())
        lh = (2 * lo.sum() * hi.sum(), 2 * (m_lo * m_hi.reindex(m_lo.index, fill_value=0)).sum())
        return dict(agree=(ll[0] + hh[0], ll[1] + hh[1]), differ=lh)

    groups = {"both rows\nhousing_loan = yes": pair_rates(housing == "yes"),
              "both rows\nhousing_loan = no": pair_rates(housing == "no")}
    groups["pooled: the rows agree\non housing_loan (what\nthe pooled test used)"] = {
        k: tuple(sum(g[k][i] for g in list(groups.values())) for i in (0, 1)) for k in ("agree", "differ")}
    lines.append("(a) share of pairs whose two rows share a month")
    for name, g in groups.items():
        a, d = g["agree"], g["differ"]
        lines.append(f"  {name.replace(chr(10), ' ')}: same pdays {a[1] / a[0]:.3f} ({a[0] / 1e6:.0f}M pairs), "
                     f"different pdays {d[1] / d[0]:.3f} ({d[0] / 1e6:.0f}M pairs)")
    for h in ("yes", "no"):
        m = housing == h
        lines.append(f"  housing_loan = {h}: {m.sum()} rows, high-pdays rows {int((m & ~low).sum())}")

    # (b) values: share of "yes" by pdays bucket, overall and within each poutcome value
    cells = []
    for name, m in [("all clients", np.ones(len(df), bool))] + [(f"poutcome =\n{v}", pout == v) for v in ("failure", "other", "success", "unknown")]:
        a, b = m & low, m & ~low
        cells.append((name, a.sum(), yes[a].mean() if a.sum() else np.nan, b.sum(), yes[b].mean() if b.sum() else np.nan))
    lines.append("(b) share of 'yes' (Class = 2)")
    for name, n_lo, y_lo, n_hi, y_hi in cells:
        lines.append(f"  {name.replace(chr(10), ' ')}: pdays -1..185: {y_lo:.3f} ({n_lo} rows) | pdays 186..871: {y_hi:.3f} ({n_hi} rows)")
    never = low & (df.V14.to_numpy() == -1)
    lines.append(f"  the low bucket holds {int(never.sum())} never-contacted rows (pdays = -1, {yes[never].mean():.3f} yes) "
                 f"and {int((low & ~never).sum())} contacted 0-185 days ago ({yes[low & ~never].mean():.3f} yes)")
    (OUT / "numbers.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.4))
    xs = np.arange(len(groups))
    for k, (key, col, lab_) in enumerate((("agree", BLUE, "rows have the same pdays bucket"), ("differ", ORANGE, "rows have different pdays buckets"))):
        vals = [g[key][1] / g[key][0] for g in groups.values()]
        ax[0].bar(xs + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=lab_)
        for x, v in zip(xs + (k - 0.5) * 0.36, vals):
            ax[0].text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax[0].set_xticks(xs, list(groups), fontsize=9)
    ax[0].set_ylabel("share of pairs whose two rows share a month")
    ax[0].set_ylim(0, 0.34)
    ax[0].set_title("(a) pdays-month given housing_loan, pairs of rows:\n"
                    "within each housing_loan value, different pdays -> fewer shared months; pooled, it flips", fontsize=11)
    ax[0].legend(fontsize=9, loc="upper right")

    xs = np.arange(len(cells))
    for k, (idx, col, lab_) in enumerate(((2, BLUE, "pdays -1..185 (never contacted, or recently)"), (4, ORANGE, "pdays 186..871 (contacted long ago)"))):
        vals = [c[idx] for c in cells]
        ax[1].bar(xs + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=lab_)
        for x, v, c in zip(xs + (k - 0.5) * 0.36, vals, cells):
            n = c[idx - 1]
            if v == v:
                ax[1].text(x, v + 0.01, f"{v:.0%}\n(n={n:,})", ha="center", fontsize=8)
    ax[1].set_xticks(xs, [c[0] for c in cells], fontsize=9)
    ax[1].set_ylabel("share of clients saying yes")
    ax[1].set_ylim(0, 0.82)
    ax[1].set_title("(b) Class vs pdays given poutcome, values:\n"
                    "overall, long-ago contact -> more yes; within every contacted group, recent contact -> more yes", fontsize=11)
    ax[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "simpson_examples.png", dpi=85)
    plt.close(fig)


if __name__ == "__main__":
    main()
