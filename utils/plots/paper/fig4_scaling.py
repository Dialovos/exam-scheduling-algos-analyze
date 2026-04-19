"""Fig 4 -- Scaling behavior by family (synthetic n=50..1000)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.plots.shared import (
    FAMILY_ORDER,
    FAMILY_COLORS,
    algo_family,
    apply_paper_style,
    load_batch018,
)


def make_fig4(out_path):
    apply_paper_style()
    b = load_batch018()
    df = b.scaling[b.scaling["feasible"] == True].copy()  # noqa: E712
    df["family"] = df["algorithm"].map(algo_family)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for fam in FAMILY_ORDER:
        fam_df = df[df["family"] == fam]
        if fam_df.empty:
            continue
        g = (fam_df.groupby("num_exams")
                    .agg(rt_mean=("runtime", "mean"), rt_std=("runtime", "std"),
                         sp_mean=("soft_penalty", "mean"), sp_std=("soft_penalty", "std"))
                    .reset_index().sort_values("num_exams"))
        color = FAMILY_COLORS.get(fam, "#888888")
        ax1.plot(g["num_exams"], g["rt_mean"], "-o", color=color, label=fam,
                 linewidth=2, markersize=6)
        ax1.fill_between(g["num_exams"],
                         (g["rt_mean"] - g["rt_std"]).clip(lower=1e-3),
                         g["rt_mean"] + g["rt_std"],
                         color=color, alpha=0.15)
        ax2.plot(g["num_exams"], g["sp_mean"], "-o", color=color, label=fam,
                 linewidth=2, markersize=6)
        ax2.fill_between(g["num_exams"],
                         (g["sp_mean"] - g["sp_std"]).clip(lower=0),
                         g["sp_mean"] + g["sp_std"],
                         color=color, alpha=0.15)

    for ax, ylabel, title, use_log_y in [
        (ax1, "Runtime (s, log)", "Runtime vs problem size", True),
        (ax2, "Soft penalty", "Soft penalty vs problem size", False),
    ]:
        ax.set_xlabel("Number of exams")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if use_log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=380)

    fig.suptitle("Scaling behavior by algorithm family",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
