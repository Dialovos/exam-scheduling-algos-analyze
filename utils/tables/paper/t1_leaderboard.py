"""T1 -- Main Leaderboard (algo x instance soft-penalty means)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.plots.shared import load_batch018, algo_order, algo_short


def _fmt_cell(mean, std):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0
    return f"{int(round(mean))} +/- {int(round(std))}"


def _instance_list(df):
    return sorted(df["dataset"].unique())


def _chain_row_from_top5(chain_top5, instances):
    row = {"Algorithm": "Chain"}
    per_ds = chain_top5.get("per_dataset_scores", {})
    for ds in instances:
        val = per_ds.get(ds)
        row[ds] = "-" if val is None else f"{int(round(val))} +/- 0"
    return row


def _ip_row(ip_soft, instances):
    row = {"Algorithm": "IP"}
    for ds in instances:
        payload = ip_soft.get(ds)
        if payload is None:
            row[ds] = "-"
        else:
            total = sum(payload.values())
            row[ds] = f"{total} +/- 0"
    return row


def make_t1(out_dir):
    """Write T1 leaderboard to ``<out_dir>/t1_leaderboard.{csv,tex}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = load_batch018()
    df = b.main.copy()
    instances = _instance_list(df)

    grouped = (df.groupby(["algorithm", "dataset"])["soft_penalty"]
                 .agg(["mean", "std"]).reset_index())

    algos_in_data = grouped["algorithm"].unique().tolist()
    ordered_algos = algo_order(algos_in_data)

    rows = []
    for algo in ordered_algos:
        row = {"Algorithm": algo_short(algo)}
        algo_df = grouped[grouped["algorithm"] == algo]
        for ds in instances:
            sub = algo_df[algo_df["dataset"] == ds]
            if sub.empty:
                row[ds] = "-"
            else:
                row[ds] = _fmt_cell(sub["mean"].iloc[0], sub["std"].iloc[0])
        rows.append(row)

    rows.append(_chain_row_from_top5(b.chain_top5, instances))
    rows.append(_ip_row(b.ip_soft, instances))

    mean_soft = (df.groupby(["algorithm", "dataset"])["soft_penalty"].mean()
                   .reset_index())
    ranks = (mean_soft.pivot(index="algorithm", columns="dataset",
                              values="soft_penalty")
                        .rank(axis=0, method="min"))
    best_on = (mean_soft.loc[
        mean_soft.groupby("dataset")["soft_penalty"].idxmin()
    ][["dataset", "algorithm"]])
    best_on_counts = best_on["algorithm"].value_counts().to_dict()

    short_to_canonical = {algo_short(a): a for a in ordered_algos}
    for row in rows:
        canonical = short_to_canonical.get(row["Algorithm"])
        if canonical is None or canonical not in ranks.index:
            row["Mean Rank"] = "-"
            row["Best On"] = "-"
        else:
            row["Mean Rank"] = f"{ranks.loc[canonical].mean():.2f}"
            row["Best On"] = str(best_on_counts.get(canonical, 0))

    columns = ["Algorithm"] + list(instances) + ["Mean Rank", "Best On"]
    out_df = pd.DataFrame(rows)[columns]

    def _rank_key(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    out_df = out_df.iloc[
        sorted(range(len(out_df)), key=lambda i: _rank_key(out_df.iloc[i]["Mean Rank"]))
    ].reset_index(drop=True)

    csv_path = out_dir / "t1_leaderboard.csv"
    tex_path = out_dir / "t1_leaderboard.tex"
    out_df.to_csv(csv_path, index=False)
    tex_path.write_text(_to_booktabs(out_df))

    return csv_path, tex_path


def _tex_escape(s):
    """Escape LaTeX special chars in a string cell (ASCII only input)."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _to_booktabs(df):
    """Hand-roll a booktabs LaTeX table. Avoids pandas' jinja2 dependency."""
    cols = list(df.columns)
    col_spec = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_tex_escape(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
