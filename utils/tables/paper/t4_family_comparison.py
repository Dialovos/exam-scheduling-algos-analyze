"""T4 -- Per-family algorithm comparison.

The fig 2 family heatmap collapses each family to its best member; this
table is the un-collapsed companion. Rows are grouped by family
(Construction, Trajectory, Population, Exact/Hybrid) and within each group
sorted by intra-family mean rank, so the reader sees both *which family
wins* and *who carries the family*.

Columns: 8 ITC instance soft-penalty means (mean +/- std), Family Rank
(mean rank within family across instances), Family Wins (count of
instances where this algo is the best in its family). The family-best
cell per (instance, family) carries a trailing ``*``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.plots.shared import (
    ALGO_FAMILY,
    FAMILY_ORDER,
    algo_short,
    load_batch018,
)


def _fmt_cell(mean, std, *, marker=""):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0
    return f"{int(round(mean))} +/- {int(round(std))}{marker}"


def make_t4(out_dir):
    """Write T4 family comparison to ``<out_dir>/t4_family_comparison.{csv,tex}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = load_batch018()
    df = b.main.copy()
    instances = sorted(df["dataset"].unique())

    grouped = (df.groupby(["algorithm", "dataset"])["soft_penalty"]
                 .agg(["mean", "std"]).reset_index())
    means = grouped.pivot(index="algorithm", columns="dataset", values="mean")
    stds = grouped.pivot(index="algorithm", columns="dataset", values="std")

    # Family-best per instance — only meaningful when the family has 2+
    # competing algos on that instance. Single-member families (Construction
    # = Greedy alone, Exact/Hybrid = CP-SAT alone in this batch) would
    # otherwise "win" trivially every row, which is misleading.
    algo_to_family = {a: ALGO_FAMILY.get(a, "Other") for a in means.index}
    fam_best = {}  # (family, dataset) -> winning algorithm
    fam_wins = {a: 0 for a in means.index}
    fam_size = {f: sum(1 for a in algo_to_family.values() if a == f)
                for f in set(algo_to_family.values())}
    for fam in FAMILY_ORDER + ["Other"]:
        fam_algos = [a for a, f in algo_to_family.items() if f == fam]
        if len(fam_algos) < 2:
            continue
        for ds in instances:
            sub = means.loc[fam_algos, ds].dropna()
            if len(sub) < 2:
                continue
            winner = sub.idxmin()
            fam_best[(fam, ds)] = winner
            fam_wins[winner] += 1

    # Intra-family rank: average rank across instances, computed per family
    fam_rank = {}
    for fam in FAMILY_ORDER + ["Other"]:
        fam_algos = [a for a, f in algo_to_family.items() if f == fam]
        if not fam_algos:
            continue
        sub = means.loc[fam_algos]
        ranks = sub.rank(axis=0, method="min")
        for a in fam_algos:
            fam_rank[a] = ranks.loc[a].mean()

    # Build rows in family-block order
    rows = []
    for fam in FAMILY_ORDER + ["Other"]:
        fam_algos = sorted(
            (a for a, f in algo_to_family.items() if f == fam),
            key=lambda a: fam_rank.get(a, float("inf")),
        )
        for a in fam_algos:
            row = {"Algorithm": algo_short(a), "Family": fam}
            for ds in instances:
                m = means.loc[a, ds] if ds in means.columns else float("nan")
                s = stds.loc[a, ds] if ds in stds.columns else float("nan")
                marker = "*" if fam_best.get((fam, ds)) == a else ""
                row[ds] = _fmt_cell(m, s, marker=marker)
            # Intra-family rank/wins are only defined when the family has
            # 2+ members; for solo-family algos we report "--" to avoid the
            # "1.00 / 8 wins" trap. (Sentinel is "--" not "n/a" so pandas
            # doesn't auto-coerce it to NaN on round-trip.)
            if fam_size.get(fam, 0) < 2:
                row["Family Rank"] = "--"
                row["Family Wins"] = "--"
            else:
                row["Family Rank"] = (f"{fam_rank[a]:.2f}"
                                      if a in fam_rank else "-")
                row["Family Wins"] = str(fam_wins.get(a, 0))
            rows.append(row)

    columns = (["Algorithm", "Family"] + list(instances)
               + ["Family Rank", "Family Wins"])
    out_df = pd.DataFrame(rows)[columns]

    csv_path = out_dir / "t4_family_comparison.csv"
    tex_path = out_dir / "t4_family_comparison.tex"
    out_df.to_csv(csv_path, index=False)
    tex_path.write_text(_to_booktabs(out_df))

    return csv_path, tex_path


def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _to_booktabs(df):
    cols = list(df.columns)
    col_spec = "ll" + "r" * (len(cols) - 2)
    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_tex_escape(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
