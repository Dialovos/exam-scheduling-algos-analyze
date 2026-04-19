"""Fig 3 -- Chain-finder methodology (2 panels: schematic + top-5 bar chart)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils.plots.shared import apply_paper_style, load_batch018


def _panel_a_schematic(ax):
    """Hand-drawn SH + prefix-cache + crossover schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    rung_y = [7.5, 5.5, 3.5]
    rung_n = [9, 3, 1]
    rung_labels = ["rung 1 (9 trials)", "rung 2 (3 trials)", "rung 3 (champion)"]
    box_color = "#4E79A7"

    for y, n, lbl in zip(rung_y, rung_n, rung_labels):
        box_w = 0.7
        total_w = n * box_w + (n - 1) * 0.1
        x0 = (10 - total_w) / 2
        for i in range(n):
            ax.add_patch(patches.Rectangle(
                (x0 + i * (box_w + 0.1), y), box_w, 0.6,
                facecolor=box_color, edgecolor="white", linewidth=0.8,
                alpha=0.85 if n > 1 else 1.0,
            ))
        ax.text(x0 - 0.3, y + 0.3, lbl, ha="right", va="center", fontsize=9)

    for y_from, y_to in zip(rung_y[:-1], rung_y[1:]):
        ax.annotate("", xy=(5, y_to + 0.7), xytext=(5, y_from - 0.1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))

    # Enlarged prefix-cache box so the 3-line label fits inside.
    box_x, box_y, box_w, box_h = 7.2, 5.2, 2.6, 1.5
    ax.add_patch(patches.FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0.08", facecolor="#FFF3E0",
        edgecolor="#F28E2B", linewidth=1.5,
    ))
    ax.text(box_x + box_w / 2, box_y + box_h / 2,
            "prefix cache\n(.sln warm-start)",
            ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.annotate("", xy=(box_x, box_y + box_h / 2),
                xytext=(6.4, box_y + box_h / 2),
                arrowprops=dict(arrowstyle="<-", lw=1.2, color="#F28E2B"))

    ax.text(5, 2.5, "1-point crossover between chains:", ha="center",
            fontsize=9, fontweight="bold")
    ax.text(5, 1.7, "[alns -> kempe | tabu]  +  [sa -> lahc | gd]", ha="center",
            fontsize=9, family="monospace")
    ax.text(5, 1.0, "->  [alns -> kempe | gd]", ha="center",
            fontsize=9, family="monospace", color="#2E7D32")

    ax.set_title("Successive Halving + Prefix Cache", fontweight="bold")


def _crop_chain(chain_steps, max_steps=4):
    parts = [step[0] for step in chain_steps]
    if len(parts) <= max_steps:
        return " -> ".join(parts)
    return " -> ".join(parts[:max_steps]) + " ..."


def _panel_b_top5_bars(ax, b):
    """Horizontal bar chart of top-5 chain scores with chain strings inside."""
    top5 = sorted(b.chain_top5["top5"], key=lambda e: e["rank"])
    total = b.chain_top5.get("total_chain_trials", 53)

    ranks = [e["rank"] for e in top5]
    scores = [e["score"] for e in top5]
    chains = [_crop_chain(e["chain"]) for e in top5]

    y = np.arange(len(top5))
    colors = ["#2E7D32" if r == 1 else "#4E79A7" for r in ranks]
    ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.invert_yaxis()

    for i, (s, c, r) in enumerate(zip(scores, chains, ranks)):
        ax.text(0.005, i, f"#{r}  {c}",
                va="center", ha="left", fontsize=9,
                color="white", fontweight="bold")
        ax.text(s - 0.005, i, f"{s:.4f}",
                va="center", ha="right", fontsize=8,
                color="white", fontweight="bold")

    ax.axvline(scores[0], color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.text(scores[0] - 0.002, -0.45, f"champion = {scores[0]:.4f}",
            color="#2E7D32", fontsize=8, fontweight="bold", ha="right", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.set_xlabel("Chain score (lower = better)")
    ax.set_xlim(0, max(scores) * 1.05)
    ax.set_title(f"Chain discovery top-5 ({total} total trials)",
                 fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)


def make_fig3(out_path):
    apply_paper_style()
    b = load_batch018()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.4]})
    _panel_a_schematic(ax1)
    _panel_b_top5_bars(ax2, b)

    fig.suptitle("Chain-finder methodology: tournament + cache + crossover",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
