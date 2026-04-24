#!/usr/bin/env python3
"""
Cross-batch comparison tables.

Pools data from the three batches under results/ and prints tabular analyses
to stdout + writes the same content to graphs/CROSS_BATCH_ANALYSIS.md.

Normalization: all quality tables use `norm_soft = soft / min(soft over all
three batches on that instance)`, so 1.0 = best achieved by anyone on that
instance. This is the only way to fairly compare batches with very different
iter budgets (batch_018 full paper budgets, batch_019 3000 iters, gpu-sweep
100 iters).

Batches:
  batch_018_colab       paper-grade, 13 base algos, 8 sets × 7 seeds (no runtime)
  batch_019_colab       Phase-2 cached/Thompson, 5 algos, 8 sets × 3 seeds
  gpu_measurement_colab Phase-3 CPU/GPU pairs, 14 algos, 5 sets × 3 seeds

Reference: smoke-test numbers for the base→cached and cached→CUDA progressions
are recorded in docs/PERF_ROADMAP.md.
"""
from __future__ import annotations

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_MD = ROOT / "graphs" / "CROSS_BATCH_ANALYSIS.md"

DISPLAY_TO_SLUG = {
    "Greedy": "greedy", "Tabu Search": "tabu", "Kempe Chain": "kempe",
    "Multi-Neighbourhood SA": "sa", "Simulated Annealing": "sa",
    "ALNS": "alns", "Great Deluge": "gd", "ABC": "abc",
    "Genetic Algorithm": "ga", "GA": "ga", "LAHC": "lahc",
    "WOA": "woa", "HHO+": "hho", "HHO": "hho",
    "CP-SAT B&B": "cpsat", "GVNS": "vns",
}
CUDA_ALIAS = {"tabu_cuda": "tabu_cached_cuda"}

FAMILIES: dict[str, list[str]] = {
    "tabu":   ["tabu", "tabu_cached", "tabu_cached_cuda"],
    "sa":     ["sa",   "sa_cached",   "sa_parallel_cuda"],
    "gd":     ["gd",   "gd_cached"],
    "lahc":   ["lahc", "lahc_cached"],
    "alns":   ["alns", "alns_thompson", "alns_cached", "alns_cuda"],
    "abc":    ["abc",  "abc_cuda"],
    "ga":     ["ga",   "ga_cuda"],
    "hho":    ["hho",  "hho_cuda"],
    "woa":    ["woa",  "woa_cuda"],
    "kempe":  ["kempe"],
    "vns":    ["vns"],
    "greedy": ["greedy"],
    "cpsat":  ["cpsat"],
}
SLUG_TO_FAMILY = {s: f for f, slugs in FAMILIES.items() for s in slugs}


def tier_of(slug: str) -> str:
    if slug.endswith("_cuda"):
        return "cuda"
    if slug.endswith("_cached") or slug == "alns_thompson":
        return "cached"
    return "base"


BATCHES = ["018 (paper)", "019 (cached)", "gpu-sweep"]
TIER_ORDER = ["base", "cached", "cuda"]

# ------------------------------------------------------------------- loaders

def load_batch_018() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "batch_018_colab" / "aggregated.csv")
    df = df.rename(columns={"algorithm": "algo_raw",
                            "dataset": "instance",
                            "soft_penalty": "soft"})
    df["algo"] = df["algo_raw"].map(DISPLAY_TO_SLUG)
    df = df.dropna(subset=["algo"])
    df["batch"] = "018 (paper)"
    df["runtime_s"] = np.nan
    return df[["batch", "algo", "instance", "seed", "soft", "runtime_s"]]


def load_batch_019() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "batch_019_colab" / "summary.csv")
    df = df.rename(columns={"runtime_sec": "runtime_s"})
    df["batch"] = "019 (cached)"
    return df[["batch", "algo", "instance", "seed", "soft", "runtime_s"]]


def load_gpu_sweep() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "gpu_measurement_colab" / "gpu_sweep_raw.csv")
    df = df.rename(columns={"set": "instance"})
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df["algo"] = df["algo"].replace(CUDA_ALIAS)
    df["batch"] = "gpu-sweep"
    return df[["batch", "algo", "instance", "seed", "soft", "runtime_s"]]


def load_all() -> pd.DataFrame:
    df = pd.concat([load_batch_018(), load_batch_019(), load_gpu_sweep()],
                   ignore_index=True)
    df["family"] = df["algo"].map(SLUG_TO_FAMILY).fillna("other")
    df["tier"] = df["algo"].apply(tier_of)
    pool_min = df.groupby("instance")["soft"].transform("min").replace(0, np.nan)
    df["norm_soft"] = df["soft"] / pool_min
    return df


# -------------------------------------------------------------- formatting

def fmt_table(df: pd.DataFrame, title: str, *, floats=2) -> str:
    out = StringIO()
    cols = list(df.columns)
    rows = df.values.tolist()
    formatted = []
    for r in rows:
        formatted.append([_fmt_cell(v, floats) for v in r])
    widths = [max(len(str(cols[i])),
                  max((len(r[i]) for r in formatted), default=0))
              for i in range(len(cols))]
    sep = "  "
    def line(cells, pad="-"):
        return sep.join(
            str(cells[i]).ljust(widths[i]) if i == 0
            else str(cells[i]).rjust(widths[i]) for i in range(len(cells))
        )
    header = line(cols)
    rule = sep.join("-" * w for w in widths)
    print(title, file=out)
    print("=" * max(len(title), len(header)), file=out)
    print(header, file=out)
    print(rule, file=out)
    for r in formatted:
        print(line(r), file=out)
    return out.getvalue()


def _fmt_cell(v, floats=2) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "–"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{floats}f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


# --------------------------------------------------------------- analyses

def algo_order(algos):
    fam_order = list(FAMILIES.keys())
    return sorted(algos,
                  key=lambda a: (fam_order.index(SLUG_TO_FAMILY.get(a, "greedy")),
                                 TIER_ORDER.index(tier_of(a)), a))


def table_coverage(df: pd.DataFrame) -> str:
    pivot = (df.groupby(["algo", "batch"])["seed"].count()
               .unstack(fill_value=0)
               .reindex(columns=BATCHES, fill_value=0))
    pivot = pivot.reindex(algo_order(pivot.index))
    pivot.insert(0, "family", [SLUG_TO_FAMILY.get(a, "other") for a in pivot.index])
    pivot.insert(1, "tier", [tier_of(a) for a in pivot.index])
    pivot = pivot.reset_index().rename(columns={"algo": "algo"})
    return fmt_table(pivot, "Table 1 — Coverage (run counts per algo × batch)")


def table_global_ranking(df: pd.DataFrame) -> str:
    agg = (df.groupby(["algo", "batch"])["norm_soft"].mean()
             .unstack().reindex(columns=BATCHES))
    agg["best"] = agg.min(axis=1)
    agg["n_batches"] = agg[BATCHES].notna().sum(axis=1)
    agg = agg.sort_values("best")
    out = agg.copy()
    out.insert(0, "family", [SLUG_TO_FAMILY.get(a, "other") for a in out.index])
    out.insert(1, "tier", [tier_of(a) for a in out.index])
    out = out.reset_index().rename(columns={"algo": "algo"})
    out = out[["algo", "family", "tier", *BATCHES, "best", "n_batches"]]
    return fmt_table(out,
        "Table 2 — Global mean normalized soft (lower = better; 1.00 = best-in-pool)")


def table_tier_progression(df: pd.DataFrame) -> str:
    tier = (df.groupby(["family", "tier"])["norm_soft"].mean()
              .unstack().reindex(columns=TIER_ORDER))
    tier["base→cached Δ"] = tier.get("cached") - tier.get("base")
    tier["cached→cuda Δ"] = tier.get("cuda") - tier.get("cached")
    tier = tier.loc[[f for f in FAMILIES if f in tier.index]]
    tier = tier.reset_index()
    return fmt_table(tier,
        "Table 3 — Tier progression by family (mean norm. soft; Δ < 0 = improvement)")


def table_instance_winners(df: pd.DataFrame) -> str:
    idx = df.groupby("instance")["soft"].idxmin()
    winners = df.loc[idx, ["instance", "algo", "batch", "soft", "seed"]]
    winners = winners.sort_values("instance").reset_index(drop=True)
    return fmt_table(winners,
        "Table 4 — Per-instance winner (minimum raw soft across every run)", floats=0)


def table_variant_delta_same_batch(df: pd.DataFrame) -> str:
    """Where a batch has both a base and a cached (or cached and cuda) variant
    of the same family, report the raw soft delta. No normalization needed —
    same batch, same iter budget, same instance set."""
    rows = []
    for fam, slugs in FAMILIES.items():
        if len(slugs) < 2:
            continue
        for b in BATCHES:
            sub = df[(df["batch"] == b) & (df["family"] == fam)]
            if sub.empty:
                continue
            means = sub.groupby("algo")["soft"].mean()
            tiers_present = {tier_of(a): a for a in means.index}
            for lhs, rhs in [("base", "cached"), ("cached", "cuda"),
                             ("base", "cuda")]:
                if lhs in tiers_present and rhs in tiers_present:
                    l_algo, r_algo = tiers_present[lhs], tiers_present[rhs]
                    l_soft, r_soft = means[l_algo], means[r_algo]
                    pct = (r_soft - l_soft) / l_soft * 100 if l_soft else np.nan
                    rows.append({
                        "family": fam, "batch": b,
                        "compare": f"{lhs} → {rhs}",
                        "lhs_algo": l_algo, "rhs_algo": r_algo,
                        "lhs_soft": l_soft, "rhs_soft": r_soft,
                        "Δ_soft": r_soft - l_soft, "Δ_%": pct,
                    })
    if not rows:
        return "Table 5 — No same-batch variant pairs available.\n"
    out = pd.DataFrame(rows)
    return fmt_table(out,
        "Table 5 — Same-batch variant deltas (Δ_% < 0 = tier upgrade improved quality)")


def table_runtime_comparison(df: pd.DataFrame) -> str:
    rt = df.dropna(subset=["runtime_s"])
    if rt.empty:
        return ""
    agg = (rt.groupby(["algo", "batch"])["runtime_s"].mean()
             .unstack().reindex(columns=BATCHES))
    agg = agg.dropna(how="all").loc[algo_order(agg.index)]
    agg = agg.reset_index().rename(columns={"algo": "algo"})
    return fmt_table(agg,
        "Table 6 — Mean runtime (s) per algo × batch (batch_018 not recorded)")


def reference_roadmap_smoke() -> str:
    """Echo the hand-measured PERF_ROADMAP numbers so the tables here sit
    next to the original smoke tests for cross-reference."""
    return """\
Table 7 — PERF_ROADMAP smoke-test reference (docs/PERF_ROADMAP.md)
==================================================================
Scope: 5000-iter single-seed smoke runs used during Phase-2 development.
These are NOT pooled with the batches above — shown for cross-reference.

algo             instance   scalar ms   cached ms   cached/scalar   soft Δ
-------------    --------   ---------   ---------   -------------   ---------------
tabu             set4            9451         787           12.0×   identical (36587)
tabu             set7            6261        1720           3.64×   identical (8653)
sa               set4             (*)         (*)           1.49×   −857 (better)
sa               set7             (*)         (*)           1.05×   −3352 (better)
gd               set4             (*)         (*)           0.82×   identical
gd               set7             (*)         (*)           1.57×   −98k (56% better)
lahc             set4             (*)         (*)           1.05×   −1826 (better)
lahc             set7             (*)         (*)           1.19×   +5% worse
alns             set4             (*)         (*)           0.75×   slight better
alns             set7             (*)         (*)           0.65×   identical
alns_thompson    set7 10k        133.3s       52.4s         2.54×   −107 (1% better)

Phase-3 CUDA smoke: see docs/PERF_ROADMAP.md §3b / §4-Option-B for
sa_parallel_cuda throughput (1.45 M iter/sec, 5.8× CPU on 64 streams).
"""


def main():
    df = load_all()
    blocks = [
        f"pooled {len(df):,} rows · {df['algo'].nunique()} algos · "
        f"{df['batch'].nunique()} batches\n",
        table_coverage(df),
        table_global_ranking(df),
        table_tier_progression(df),
        table_instance_winners(df),
        table_variant_delta_same_batch(df),
        table_runtime_comparison(df),
        reference_roadmap_smoke(),
    ]
    combined = "\n".join(blocks)
    print(combined)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("# Cross-batch analysis\n\n"
                      "Auto-generated by `scripts/make_batch_comparison.py`.\n\n"
                      "```\n" + combined + "```\n")
    print(f"\nsaved -> {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
