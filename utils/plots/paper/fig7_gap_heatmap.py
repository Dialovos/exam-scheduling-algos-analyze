"""Fig 7 -- Gap-to-IP heatmap.

Rows = algorithms sorted by mean gap, cols = ITC instances where CP-SAT
found a usable optimum. Cell = (mean algo soft / IP soft - 1) * 100.

``exam_comp_set8`` is excluded because the CP-SAT optimum on that instance
(IP_soft = 25) does not share a common evaluator scale with the heuristic
pipeline (heuristics score 12k+ there), so the ratio is not meaningful.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.plots.shared import algo_short, apply_paper_style, load_batch018


EXCLUDE_INSTANCES = {"exam_comp_set8"}
GAP_DISPLAY_CAP = 300.0  # text shows ">300%" above this


def _ip_total(sd):
    return sum(sd.values())


def _fmt_cell(val):
    if np.isnan(val):
        return ""
    if val >= GAP_DISPLAY_CAP:
        return f">{int(GAP_DISPLAY_CAP)}%"
    if val <= -10:
        return f"{val:.0f}%"
    return f"{val:.1f}%" if abs(val) < 10 else f"{val:.0f}%"


def make_fig7(out_path):
    apply_paper_style()
    b = load_batch018()

    ip_totals = {
        inst: _ip_total(sd) for inst, sd in b.ip_soft.items()
        if inst not in EXCLUDE_INSTANCES
    }
    solved = sorted(ip_totals.keys())

    df = b.main[b.main["dataset"].isin(solved)].copy()
    per_algo = df.groupby(["algorithm", "dataset"])["soft_penalty"].mean().reset_index()

    gap_rows = {}
    for _, row in per_algo.iterrows():
        algo = row["algorithm"]
        inst = row["dataset"]
        ip_val = ip_totals[inst]
        if ip_val <= 0:
            continue
        gap = (row["soft_penalty"] / ip_val - 1) * 100
        gap_rows.setdefault(algo, {})[inst] = gap

    algos = sorted(gap_rows.keys(),
                   key=lambda a: np.mean(list(gap_rows[a].values())))
    matrix = np.full((len(algos), len(solved)), np.nan)
    for i, algo in enumerate(algos):
        for j, inst in enumerate(solved):
            matrix[i, j] = gap_rows[algo].get(inst, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    vmax = GAP_DISPLAY_CAP
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=-20, vmax=vmax)

    ax.set_xticks(range(len(solved)))
    ax.set_xticklabels([s.replace("exam_comp_", "") for s in solved])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([algo_short(a) for a in algos])

    mid = vmax / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, _fmt_cell(val),
                    ha="center", va="center", fontsize=8,
                    color="white" if val > mid or val < -10 else "black")

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if np.all(np.isnan(col)):
            continue
        i_min = int(np.nanargmin(col))
        ax.add_patch(Rectangle((j - 0.5, i_min - 0.5), 1, 1,
                               fill=False, edgecolor="#2E7D32", linewidth=2.4))

    ax.set_title(
        "Gap to CP-SAT IP optimum per instance\n"
        "(cell = (algo_soft / ip_soft - 1) * 100%; algos sorted by mean gap; "
        "green box = best per instance;\n"
        "set8 excluded - evaluator-scale mismatch; cells > 300% capped in display)",
        fontweight="bold", fontsize=9)
    fig.colorbar(im, ax=ax, label="Gap to IP (%, capped at 300)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
