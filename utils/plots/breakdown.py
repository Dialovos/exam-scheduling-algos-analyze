"""Soft-constraint breakdown + 2x2 dashboard figures.

Any plot whose job is to decompose a single algorithm's (or every algorithm's)
soft penalty into its seven constituent components belongs here. The
dashboard is included because it's a composite of breakdown + comparison +
feasibility rendered into a single figure.
"""
from __future__ import annotations

import os

import numpy as np

from utils.plots.shared import (
    SOFT_KEYS, SOFT_LABELS, SOFT_COLORS, HAS_MPL,
    _c, _short, _order, _style, _kfmt, _apply_xlabels, _save,
)
from utils.plots.comparative import plot_algorithm_comparison

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:  # pragma: no cover
    pass


def plot_soft_breakdown(df_or_dict, dataset=None, title=None, save_path=None):
    """Stacked bars — one bar per algorithm, seven segments each.

    Switches to horizontal stacks at >10 algorithms: labels don't rotate,
    totals land at the bar ends, and the legend moves below the plot so
    it no longer collides with tall bars (the old vertical layout had the
    legend box covering the tallest algorithm's total at n=13).
    """
    if not HAS_MPL:
        return
    _style()
    if isinstance(df_or_dict, pd.DataFrame):
        data = df_or_dict[df_or_dict["dataset"] == dataset] if dataset else df_or_dict
        breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS}
                     for a, g in data.groupby("algorithm")}
    else:
        breakdown = df_or_dict

    algos = _order(breakdown.keys())
    n = len(algos)
    horizontal = n > 10

    if horizontal:
        fig, ax = plt.subplots(figsize=(11, max(4, n * 0.55)))
        y = np.arange(n)
        lefts = np.zeros(n)
        bh = 0.72
        for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
            vals = np.array([breakdown[a].get(key, 0) for a in algos])
            ax.barh(y, vals, left=lefts, label=label, color=color,
                    alpha=0.90, height=bh, edgecolor="white", linewidth=0.3)
            lefts += vals

        right = lefts.max() if len(lefts) else 1
        for j in range(n):
            ax.text(lefts[j] + right * 0.01, j, f"{lefts[j]:,.0f}",
                    ha="left", va="center", fontsize=9, fontweight="bold")
        _kfmt(ax, axis="x")
        ax.set_yticks(y)
        ax.set_yticklabels([_short(a) for a in algos], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, right * 1.18)
        ax.set_xlabel("Soft Penalty")
        ax.set_title(title or f"Soft Constraint Breakdown"
                     f"{' — ' + dataset if dataset else ''}", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=min(len(SOFT_KEYS), 7), fontsize=9, frameon=False)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 6))
        x = np.arange(n)
        bottoms = np.zeros(n)
        bw = min(0.7, 5.0 / max(n, 1))

        for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
            vals = np.array([breakdown[a].get(key, 0) for a in algos])
            ax.bar(x, vals, bottom=bottoms, label=label, color=color,
                   alpha=0.90, width=bw, edgecolor="white", linewidth=0.3)
            bottoms += vals

        top = bottoms.max() if len(bottoms) else 1
        for j in range(n):
            ax.text(j, bottoms[j] + top * 0.015, f"{bottoms[j]:,.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        _apply_xlabels(ax, algos)
        _kfmt(ax)
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel("Soft Penalty")
        ax.set_title(title or f"Soft Constraint Breakdown"
                     f"{' — ' + dataset if dataset else ''}", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  ncol=min(len(SOFT_KEYS), 7), fontsize=9, frameon=False)
        fig.tight_layout()

    _save(fig, save_path)
    return fig


def plot_soft_lines(df, dataset=None, title=None, save_path=None):
    """Line chart — x = algorithm, y = soft component value, one line per
    component. Good for spotting which penalties dominate each algorithm."""
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df.copy()
    means = data.groupby("algorithm")[SOFT_KEYS].mean()
    algos = _order(means.index)
    means = means.reindex(algos)

    fig, ax = plt.subplots(figsize=(max(8, len(algos) * 1.2), 6))
    x = np.arange(len(algos))

    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = means[key].values
        ax.plot(x, vals, "o-", label=label, color=color,
                linewidth=2, markersize=7, alpha=0.85)

    _apply_xlabels(ax, algos)
    _kfmt(ax)
    ax.set_ylabel("Penalty Value")
    ax.set_title(title or f"Soft Components by Algorithm"
                 f"{' — ' + dataset if dataset else ''}", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_summary_dashboard(df, dataset=None, save_path=None):
    """2x2 dashboard: soft bars, runtime bars, breakdown, feasibility."""
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df
    algos = _order(data["algorithm"].unique())
    n = len(algos)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_algorithm_comparison(df, dataset=dataset, metric="soft_penalty", ax=axes[0, 0])
    plot_algorithm_comparison(df, dataset=dataset, metric="runtime", ax=axes[0, 1])

    ax = axes[1, 0]
    breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS}
                 for a, g in data.groupby("algorithm")}
    x = np.arange(n)
    bottoms = np.zeros(n)
    bw = min(0.7, 5.0 / max(n, 1))
    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = np.array([breakdown.get(a, {}).get(key, 0) for a in algos])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color,
               alpha=0.90, width=bw, edgecolor="white", linewidth=0.3)
        bottoms += vals
    _apply_xlabels(ax, algos)
    _kfmt(ax)
    top = bottoms.max() if len(bottoms) else 1
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("Soft Penalty")
    ax.set_title("Soft Constraint Breakdown", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    ax = axes[1, 1]
    rates = data.groupby("algorithm")["feasible"].mean().reindex(algos).fillna(0) * 100
    y = np.arange(n)
    ax.barh(y, rates.values, color=[_c(a) for a in algos],
            alpha=0.88, height=0.55, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(rates.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in algos], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("Feasibility (%)")
    ax.set_title("Feasibility Rate", fontweight="bold")

    ds_label = f" — {dataset}" if dataset else ""
    fig.suptitle(f"Results Dashboard{ds_label}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)
    return fig


def plot_soft_constraint_breakdown(breakdown: dict, output_dir: str = "results"):
    """Legacy interface for main.py — thin wrapper around ``plot_soft_breakdown``."""
    plot_soft_breakdown(breakdown,
                        save_path=os.path.join(output_dir, "soft_constraint_breakdown.png"))
