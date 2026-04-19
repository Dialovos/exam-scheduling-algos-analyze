"""Fig 5 -- Parameter sensitivity fingerprint (split view).

Two panels:
  (a) Horizontal bar chart of *iters* sensitivity per algo -- sorted descending.
      This is the universal sweep param: every algo has it, so it gets a
      dedicated panel that doubles as a per-algo "tuning urgency" ranking.
  (b) Compact heatmap of *non-iters* params (pop, list, patience, tenure,
      budget) -- only the algos/columns that actually swept these are shown,
      avoiding the sparse-grid problem of the previous single-panel design.

Sensitivity = (max - min) / mean of soft_penalty across that param's sweep
values, with feasible-only rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

from utils.plots.shared import (
    ALGO_FAMILY,
    FAMILY_COLORS,
    algo_short,
    apply_paper_style,
    load_batch018,
)

ALGO_CANONICAL = {
    "abc": "ABC", "alns": "ALNS", "ga": "Genetic Algorithm",
    "gd": "Great Deluge", "kempe": "Kempe Chain", "lahc": "LAHC",
    "sa": "Simulated Annealing", "tabu": "Tabu Search",
    "vns": "GVNS", "woa": "WOA", "hho": "HHO+",
}

_PARAM_PREFIXES = ("abc_", "alns_", "ga_", "gd_", "kempe_", "lahc_", "sa_",
                   "tabu_", "vns_", "woa_", "hho_")


def _canonical_param(p):
    for prefix in _PARAM_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def _sensitivity(sub):
    per_val = sub.groupby("param_value")["soft_penalty"].mean()
    if len(per_val) < 2:
        return np.nan
    sp_mean = per_val.mean()
    if sp_mean == 0:
        return np.nan
    return float((per_val.max() - per_val.min()) / sp_mean)


def make_fig5(out_path):
    apply_paper_style()
    b = load_batch018()
    df = b.sweep[b.sweep["feasible"] == True].copy()  # noqa: E712
    df["algorithm"] = df["algorithm"].map(lambda a: ALGO_CANONICAL.get(a, a))
    df["param_canonical"] = df["param_col"].map(_canonical_param)

    # ── Panel A: iters bars ──────────────────────────────────────────────
    iters_df = df[df["param_canonical"] == "iters"]
    iters_rows = []
    for algo, sub in iters_df.groupby("algorithm"):
        s = _sensitivity(sub)
        if not np.isnan(s):
            iters_rows.append((algo, s))
    iters_rows.sort(key=lambda r: r[1], reverse=True)
    algos_iters = [r[0] for r in iters_rows]
    sens_iters = np.array([r[1] for r in iters_rows])
    colors_iters = [FAMILY_COLORS.get(ALGO_FAMILY.get(a, ""), "#888")
                    for a in algos_iters]

    # ── Panel B: non-iters heatmap (only populated cells) ────────────────
    other_params = sorted(p for p in df["param_canonical"].unique() if p != "iters")
    other_algos_set = set()
    for p in other_params:
        other_algos_set.update(df[df["param_canonical"] == p]["algorithm"].unique())
    # Order by iters-sensitivity rank when possible, then alphabetical
    iters_rank = {a: i for i, a in enumerate(algos_iters)}
    other_algos = sorted(other_algos_set,
                         key=lambda a: (iters_rank.get(a, 99), a))

    matrix = np.full((len(other_algos), len(other_params)), np.nan)
    for i, algo in enumerate(other_algos):
        for j, param in enumerate(other_params):
            sub = df[(df["algorithm"] == algo) & (df["param_canonical"] == param)]
            if sub.empty:
                continue
            s = _sensitivity(sub)
            if not np.isnan(s):
                matrix[i, j] = s

    # ── Layout ───────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.32},
    )

    # Panel A: bars
    y = np.arange(len(algos_iters))
    axA.barh(y, sens_iters, color=colors_iters,
             edgecolor="white", linewidth=0.7)
    axA.set_yticks(y)
    axA.set_yticklabels([algo_short(a) for a in algos_iters])
    axA.invert_yaxis()
    axA.set_xlabel("Sensitivity to iters  =  (max - min) / mean of soft")
    axA.set_title("(a) Iter-budget sensitivity (universal param)",
                  fontweight="bold", fontsize=11)
    axA.grid(True, axis="x", alpha=0.3)
    axA.set_axisbelow(True)
    for i, v in enumerate(sens_iters):
        axA.text(v + 0.03, i, f"{v:.2f}", va="center", ha="left",
                 fontsize=9, fontweight="bold")
    axA.set_xlim(0, max(sens_iters.max() * 1.18, 0.5))

    # Panel B: non-iters heatmap
    if matrix.size and not np.all(np.isnan(matrix)):
        vmax = float(np.nanmax(matrix))
    else:
        vmax = 1.0
    im = axB.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    axB.set_xticks(range(len(other_params)))
    axB.set_xticklabels(other_params, rotation=20, ha="right")
    axB.set_yticks(range(len(other_algos)))
    axB.set_yticklabels([algo_short(a) for a in other_algos])
    axB.set_title("(b) Non-iter param sensitivity\n(blank = param not swept)",
                  fontweight="bold", fontsize=11)

    text_stroke = [pe.withStroke(linewidth=2.2, foreground="black")]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                axB.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center",
                         fontsize=9, fontweight="bold",
                         color="white", path_effects=text_stroke)

    # Red border on top-1 non-iters param per algo
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        if np.all(np.isnan(row)):
            continue
        j_max = int(np.nanargmax(row))
        axB.add_patch(Rectangle((j_max - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="red", linewidth=2.2))

    fig.colorbar(im, ax=axB, label="Sensitivity (dimensionless)",
                 fraction=0.046, pad=0.04)

    fig.suptitle("Parameter sensitivity fingerprint  "
                 "(red box = top-1 non-iter param per algo; "
                 "bar color = algo family)",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
