"""Fig 6 -- IP vs best heuristic, stacked soft components per instance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.plots.shared import (
    SOFT_KEYS,
    SOFT_LABELS,
    SOFT_COLORS,
    algo_short,
    apply_paper_style,
    load_batch018,
)


def _best_heuristic_breakdown(main_df, dataset):
    sub = main_df[main_df["dataset"] == dataset]
    if sub.empty:
        return None, None
    mean_by_algo = (sub.groupby("algorithm")[SOFT_KEYS + ["soft_penalty"]]
                       .mean().reset_index())
    best = mean_by_algo.loc[mean_by_algo["soft_penalty"].idxmin()]
    return best["algorithm"], {k: float(best[k]) for k in SOFT_KEYS}


def make_fig6(out_path):
    apply_paper_style()
    b = load_batch018()
    datasets = sorted(b.ip_soft.keys())

    ip_stacks = {k: np.zeros(len(datasets)) for k in SOFT_KEYS}
    heur_stacks = {k: np.zeros(len(datasets)) for k in SOFT_KEYS}
    heur_labels = []

    for i, ds in enumerate(datasets):
        for k in SOFT_KEYS:
            ip_stacks[k][i] = b.ip_soft[ds].get(k, 0)
        algo, breakdown = _best_heuristic_breakdown(b.main, ds)
        heur_labels.append(algo_short(algo) if algo else "-")
        if breakdown:
            for k in SOFT_KEYS:
                heur_stacks[k][i] = breakdown[k]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(datasets))
    w = 0.38
    bot_ip = np.zeros(len(datasets))
    bot_heur = np.zeros(len(datasets))

    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        ax.bar(x - w / 2, ip_stacks[key], w, bottom=bot_ip, color=color,
               label=label, edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, heur_stacks[key], w, bottom=bot_heur, color=color,
               edgecolor="white", linewidth=0.5)
        bot_ip += ip_stacks[key]
        bot_heur += heur_stacks[key]

    ymax = max(bot_ip.max(), bot_heur.max())
    for i in range(len(datasets)):
        ax.text(x[i] - w / 2, bot_ip[i], f"{int(bot_ip[i]):,}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(x[i] + w / 2, bot_heur[i], f"{int(bot_heur[i]):,}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(x[i] + w / 2, -ymax * 0.05, f"best: {heur_labels[i]}",
                ha="center", va="top", fontsize=8, color="#555")
        if bot_heur[i] > 0:
            gap = (bot_ip[i] - bot_heur[i]) / bot_heur[i] * 100
            ax.text(x[i], ymax * 1.08, f"{gap:+.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#777",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("exam_comp_", "") for d in datasets])
    ax.set_xlabel("ITC 2007 instance (solved by IP within 2h)")
    ax.set_ylabel("Soft penalty (stacked by component)")
    ax.set_title("CP-SAT IP vs best heuristic, per instance", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0.0, fontsize=8, framealpha=0.92,
              title="Soft component", title_fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-ymax * 0.1, ymax * 1.18)

    fig.text(0.12, 0.01,
             "Left bar: CP-SAT IP  |  Right bar: best heuristic.  "
             "Sets 3, 5, 7 excluded (IP timeout > 2h).",
             fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(Path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
