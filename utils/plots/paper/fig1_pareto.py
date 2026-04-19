"""Fig 1 -- Pareto hero (synthetic n=1000, all algos x 3 seeds).

The main ITC batch lacks runtime; this figure uses the scaling batch's
largest size so the Pareto story is honest and complete.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.plots.shared import (
    FAMILY_COLORS,
    FAMILY_ORDER,
    algo_family,
    algo_short,
    apply_paper_style,
    load_batch018,
)


def _pareto_front(points):
    """Indices of non-dominated points in (x, y); lower is better on both."""
    pts = np.asarray(points)
    order = np.argsort(pts[:, 0])
    front = []
    best_y = np.inf
    for idx in order:
        if pts[idx, 1] < best_y:
            front.append(idx)
            best_y = pts[idx, 1]
    return front


def make_fig1(out_path):
    apply_paper_style()
    b = load_batch018()
    df = b.scaling[(b.scaling["num_exams"] == 1000) &
                   (b.scaling["feasible"] == True)].copy()  # noqa: E712

    if df.empty:
        raise RuntimeError("Fig 1 requires scaling batch rows at num_exams=1000")

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, sub in df.groupby("algorithm"):
        fam = algo_family(algo)
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter(sub["runtime"], sub["soft_penalty"],
                   color=color, alpha=0.35, s=30, edgecolor="none", zorder=1)

    agg = (df.groupby("algorithm")
             .agg(rt_mean=("runtime", "mean"), rt_std=("runtime", "std"),
                  sp_mean=("soft_penalty", "mean"), sp_std=("soft_penalty", "std"))
             .reset_index())

    for _, row in agg.iterrows():
        fam = algo_family(row["algorithm"])
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.errorbar(row["rt_mean"], row["sp_mean"],
                    xerr=row["rt_std"], yerr=row["sp_std"],
                    fmt="o", color=color, markersize=11,
                    markeredgecolor="white", markeredgewidth=1.5,
                    ecolor=color, elinewidth=1.2, capsize=3, alpha=0.95,
                    zorder=3)
        ax.text(row["rt_mean"] * 1.08, row["sp_mean"],
                algo_short(row["algorithm"]), fontsize=9, va="center", zorder=4)

    pts = agg[["rt_mean", "sp_mean"]].values
    front_idx = _pareto_front(pts)
    front_idx_sorted = sorted(front_idx, key=lambda i: pts[i, 0])
    front_pts = pts[front_idx_sorted]
    ax.plot(front_pts[:, 0], front_pts[:, 1], "--", color="#555555",
            linewidth=1.6, alpha=0.8, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s, log scale)")
    ax.set_ylabel("Soft penalty")
    ax.set_title("Pareto frontier: quality vs runtime (synthetic, n=1000 exams)",
                 fontweight="bold")

    seen_families = [f for f in FAMILY_ORDER
                     if f in {algo_family(a) for a in agg["algorithm"]}]
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=FAMILY_COLORS.get(f, "#888"),
                                  markersize=10, label=f)
                       for f in seen_families]
    legend_handles.append(plt.Line2D([0], [0], color="#555555", linestyle="--",
                                     label="Pareto frontier"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.text(0.99, 0.02,
            "Construction (Greedy) infeasible at n=1000; IP + champion "
            "chain on ITC sets: see Fig 6 and Table 2",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#555",
            bbox=dict(facecolor="white", edgecolor="#DDD",
                      boxstyle="round,pad=0.3"))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
