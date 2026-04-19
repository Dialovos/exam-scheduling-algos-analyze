"""Fig 2 -- Family Dominance Heatmap.

Rows = 8 ITC instances, cols = 4 families (Construction, Trajectory,
Population, Exact / Hybrid). Cell = best-in-family normalized soft
penalty. Bold border on the family containing the instance winner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.plots.shared import (
    FAMILY_ORDER,
    algo_family,
    algo_short,
    apply_paper_style,
    load_batch018,
    normalize_per_instance,
)


def make_fig2(out_path):
    apply_paper_style()
    b = load_batch018()
    df = normalize_per_instance(b.main)

    instances = sorted(df["dataset"].unique())
    families = FAMILY_ORDER

    value = np.full((len(instances), len(families)), np.nan)
    annot = np.full(value.shape, "", dtype=object)

    for i, ds in enumerate(instances):
        ds_rows = df[df["dataset"] == ds]
        for j, fam in enumerate(families):
            fam_rows = ds_rows[ds_rows["algorithm"].map(algo_family) == fam]
            if fam_rows.empty:
                continue
            mean_by_algo = (fam_rows.groupby("algorithm")["soft_norm"]
                                     .mean().reset_index())
            best_idx = mean_by_algo["soft_norm"].idxmin()
            best_algo = mean_by_algo.loc[best_idx, "algorithm"]
            best_val = mean_by_algo.loc[best_idx, "soft_norm"]
            value[i, j] = best_val
            annot[i, j] = f"{algo_short(best_algo)}\n{best_val:.2f}"

    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmax_data = np.nanmax(value) if np.any(np.isfinite(value)) else 1.5
    im = ax.imshow(value, cmap="RdYlGn_r", aspect="auto",
                   vmin=1.0, vmax=min(float(vmax_data), 1.5))

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels([d.replace("exam_comp_", "") for d in instances])

    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            if annot[i, j]:
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=9, color="black")

    for i in range(value.shape[0]):
        row = value[i, :]
        if np.all(np.isnan(row)):
            continue
        j_winner = int(np.nanargmin(row))
        ax.add_patch(Rectangle((j_winner - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="black", linewidth=2.2))

    ax.set_title("Family dominance per ITC 2007 instance\n"
                 "(cell = best-in-family normalized soft; 1.0 = instance winner)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="Normalized soft penalty (lower is better)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
