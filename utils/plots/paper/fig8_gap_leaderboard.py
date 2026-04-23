"""Fig 8 -- Gap-to-IP leaderboard.

Horizontal bars, mean gap to CP-SAT IP optimum across 4 solved ITC
instances. Error bar = std across instances. Bars colored by family.
IP optimum marked at x = 0.

Two compute budgets are in play and they are NOT the same solver run:

* **Baseline (x = 0)**: the ``ip`` algorithm in ``colab_batch_ip/`` ---
  CP-SAT with a Tabu warm-start hint (``--ip-warmstart``), a 2 h wall-clock
  cap, and ``--ip-workers 0`` (all cores). On the 4 sets in this figure
  (set1, set2, set4, set6) it proves optimum well under the 2 h cap. The
  Tabu pre-compute time is not folded into the baseline cost; see
  PROGRESS.md (2026-04-22 note) for the accounting caveat.

* **"CP-SAT (60s)" bar**: the ``cpsat`` algorithm in the main batch,
  cold (no warm-start), 60 s time budget per seed, 7 seeds per set. The
  positive gap reflects the shorter budget and cold start --- not a defect
  of the solver.

``exam_comp_set8`` is excluded because the baseline optimum on that
instance (IP_soft = 25) does not share a common evaluator scale with the
heuristic pipeline (heuristics score 12k+ there), so the ratio is not
meaningful.

The x-axis is symlog so single-digit % gaps and 1000%+ failures are both
visible on one chart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.plots.shared import (
    ALGO_FAMILY,
    FAMILY_COLORS,
    FAMILY_ORDER,
    algo_short,
    apply_paper_style,
    load_batch018,
)


EXCLUDE_INSTANCES = {"exam_comp_set8"}


def _ip_total(sd):
    return sum(sd.values())


def make_fig8(out_path):
    apply_paper_style()
    b = load_batch018()

    ip_totals = {
        inst: _ip_total(sd) for inst, sd in b.ip_soft.items()
        if inst not in EXCLUDE_INSTANCES
    }
    solved = sorted(ip_totals.keys())

    df = b.main[b.main["dataset"].isin(solved)].copy()
    per_algo_inst = df.groupby(["algorithm", "dataset"])["soft_penalty"].mean().reset_index()

    rows = []
    for _, r in per_algo_inst.iterrows():
        inst = r["dataset"]
        ip_val = ip_totals[inst]
        if ip_val <= 0:
            continue
        gap = (r["soft_penalty"] / ip_val - 1) * 100
        rows.append({"algorithm": r["algorithm"], "dataset": inst, "gap_pct": gap})

    gap_df = pd.DataFrame(rows)
    agg = (gap_df.groupby("algorithm")["gap_pct"]
                 .agg(mean="mean", std="std").reset_index()
                 .sort_values("mean", ascending=True)
                 .reset_index(drop=True))
    agg["std"] = agg["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(agg))
    colors = [FAMILY_COLORS.get(ALGO_FAMILY.get(a, "Trajectory"), "#777777")
              for a in agg["algorithm"]]
    ax.barh(y, agg["mean"], color=colors,
            edgecolor="white", linewidth=0.6)

    ax.set_xscale("symlog", linthresh=10)

    x_max_val = float(agg["mean"].max())
    x_min_val = float(agg["mean"].min())
    right_edge = x_max_val * 1.8
    left_edge = min(x_min_val * 1.2, -5)

    for i, row in agg.iterrows():
        val = row["mean"]
        x_text = val + max(abs(val) * 0.08, 2) if val >= 0 else val - 2
        label = f"{val:+.0f}%"
        ha = "left" if val >= 0 else "right"
        ax.text(x_text, i, label, va="center", ha=ha,
                fontsize=8, fontweight="bold")

    def _disambig(name):
        return "CP-SAT (60s)" if algo_short(name) == "CP-SAT" else algo_short(name)

    ax.set_yticks(y)
    ax.set_yticklabels([_disambig(a) for a in agg["algorithm"]])
    ax.invert_yaxis()
    ax.axvline(0, color="#2E7D32", linewidth=2, linestyle="--")

    ax.set_xlabel("Mean gap to CP-SAT IP optimum (%, symlog)")
    ax.set_title("Leaderboard vs CP-SAT IP optimum\n"
                 "(Baseline = IP run: CP-SAT with Tabu warm-start, 2 h wall-clock cap, "
                 "proved optimum on these 4 sets. 'CP-SAT (60s)' = cold CP-SAT in the "
                 "main batch, 60 s per seed. Set8 excluded - evaluator-scale mismatch)",
                 fontweight="bold", fontsize=9.5)

    handles = [Patch(facecolor=FAMILY_COLORS[f], label=f) for f in FAMILY_ORDER]
    handles.append(plt.Line2D([], [], color="#2E7D32", linestyle="--",
                              linewidth=2, label="IP optimum (2h cap, proved)"))
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
              fontsize=9, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3, which="both")

    ax.set_xlim(left=left_edge, right=right_edge)

    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
