"""
Case 4 of mediation_cases.py in detail: bank, edge pdays - month, removed by V2 as
explained by housing_loan. Shows, from all row pairs of the full data, how often two rows
share a month, split by housing_loan (yes-yes / no-no pairs) and by whether the rows agree
on pdays, next to what V2's test sees (yes-yes and no-no pooled as "agree on housing_loan").

Run with:  python explorations/case4_simpson.py
Writes results/jar_vs_two_sided/cases/4_detail_simpson.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diada import paths  # noqa: E402
from diada.core.diada_tool import _label_columns  # noqa: E402

OUT = paths.RESULTS_DIR / "jar_vs_two_sided" / "cases"


def main() -> None:
    df = pd.read_csv(paths.DATA_DIR / "base_datasets" / "bank.csv")
    lab = _label_columns(df.copy(), 10)
    lab.columns = df.columns
    pdays = lab.V14.map(lambda v: "low" if v.left < 0 else "high").to_numpy()
    housing, month = lab.V7.to_numpy(), lab.V11.astype(str).to_numpy()

    def count(rows1, rows2):
        """(ordered pairs i != j with i in rows1, j in rows2, of those sharing a month)."""
        m1, m2 = pd.Series(month[rows1]).value_counts(), pd.Series(month[rows2]).value_counts()
        same = (m1 * m2.reindex(m1.index, fill_value=0)).sum()
        if rows1 is rows2:
            return rows1.sum() ** 2 - rows1.sum(), same - rows1.sum()
        return 2 * rows1.sum() * rows2.sum(), 2 * same

    groups = {}
    for h in ("yes", "no"):
        lo, hi = (housing == h) & (pdays == "low"), (housing == h) & (pdays == "high")
        ll, hh, lh = count(lo, lo), count(hi, hi), count(lo, hi)
        groups[f"both housing_loan = {h}"] = dict(agree=(ll[0] + hh[0], ll[1] + hh[1]), differ=lh)
    pooled = {k: tuple(sum(g[k][i] for g in groups.values()) for i in (0, 1)) for k in ("agree", "differ")}
    groups["pooled: agree on housing_loan\n(what V2 tests)"] = pooled

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, g) in enumerate(groups.items()):
        for k, (key, color) in enumerate((("agree", "#378ADD"), ("differ", "#D85A30"))):
            n, same = g[key]
            x = i * 3 + k
            ax.bar(x, same / n, color=color, label=f"rows agree on pdays" if key == "agree" else "rows differ on pdays")
            ax.text(x, same / n, f"{same / n:.3f}\n{n / 1e6:,.0f}M pairs", ha="center", va="bottom", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper right")
    ax.set_xticks([0.5, 3.5, 6.5], list(groups))
    ax.set_ylabel("share of pairs whose two rows share a month")
    ax.set_ylim(0, 0.36)
    ax.set_title("bank, pdays - month given housing_loan: within each housing_loan value, rows that differ on pdays\n"
                 "share a month less often; pooled, the difference reverses (Simpson's paradox)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "4_detail_simpson.png", dpi=85)
    for name, g in groups.items():
        print(name.replace("\n", " "), {k: (int(n), round(s / n, 4)) for k, (n, s) in g.items()})


if __name__ == "__main__":
    main()
