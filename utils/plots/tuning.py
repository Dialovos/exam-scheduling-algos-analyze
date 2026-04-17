"""Parameter-sensitivity plots for the auto-tuner output.

Each function here takes a long-form DataFrame whose rows are individual
tuning trials and produces a single-algo sensitivity figure. Multi-algo
comparison plots do NOT belong here — those live in
:mod:`utils.plots.comparative`.

Two flavours:
    * ``plot_parameter_sensitivity`` — 1-D dual-axis curve (soft ± std, runtime ± std)
    * ``plot_tuning_sensitivity``    — 2-D heatmap when a sweep covers two knobs
"""
from __future__ import annotations

import numpy as np

from utils.plots.shared import METRIC_LABELS, HAS_MPL, _style, _kfmt, _save

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    pass


def plot_parameter_sensitivity(df, param_col, metric="soft_penalty",
                               algorithm=None, title=None, save_path=None):
    """Dual-axis sensitivity plot for a single parameter.

    Left axis: ``metric`` mean ±std; right axis: runtime mean ±std.
    Filter by *algorithm* when the DataFrame pools multiple algos together.
    """
    if not HAS_MPL:
        return
    _style()
    data = df[df["algorithm"] == algorithm] if algorithm else df.copy()
    g = (data.groupby(param_col)
         .agg(soft_mean=(metric, "mean"), soft_std=(metric, "std"),
              rt_mean=("runtime", "mean"), rt_std=("runtime", "std"))
         .reset_index().sort_values(param_col))
    if g.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    c1, c2 = "#59A14F", "#E15759"
    ax1.errorbar(g[param_col], g["soft_mean"], yerr=g["soft_std"],
                 color=c1, marker="^", linewidth=2.2, capsize=4,
                 label=METRIC_LABELS.get(metric, metric))
    ax1.set_xlabel(param_col.replace("_", " ").title())
    ax1.set_ylabel(METRIC_LABELS.get(metric, metric), color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    _kfmt(ax1)

    ax2 = ax1.twinx()
    ax2.errorbar(g[param_col], g["rt_mean"], yerr=g["rt_std"],
                 color=c2, marker="s", linewidth=2, capsize=4,
                 linestyle="--", label="Runtime")
    ax2.set_ylabel("Runtime (s)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.spines["right"].set_visible(True)

    al = f" ({algorithm})" if algorithm else ""
    ax1.set_title(title or f"Parameter Sensitivity{al}", fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_tuning_sensitivity(sens_df, algorithm=None, metric="soft_penalty",
                            title=None, save_path=None):
    """Heatmap: two-parameter sensitivity grid for one algo.

    Expects a long-form DataFrame with columns ``param_a``, ``value_a``,
    ``param_b``, ``value_b``, plus *metric* (default ``soft_penalty``).
    The grid is aggregated by mean so you can pass multi-seed raw
    measurements directly — the heatmap averages them.

    Degrades to a 1-D curve when the algo only has one tunable knob
    (``param_b`` is empty/NaN throughout): calls
    :func:`plot_parameter_sensitivity` on the same frame so the caller
    gets *something* useful either way.
    """
    if not HAS_MPL:
        return
    df = sens_df[sens_df["algorithm"] == algorithm] if algorithm else sens_df.copy()
    if df.empty:
        print(f"[plot_tuning_sensitivity] no data for {algorithm!r}")
        return

    # 1-D fallback: param_b unset for every row.
    has_b = "param_b" in df.columns and df["param_b"].astype(str).str.len().gt(0).any()
    if not has_b:
        # Normalise column names for plot_parameter_sensitivity
        tmp = df.rename(columns={"value_a": "param_value"})
        tmp["param_col"] = df["param_a"]
        param_name = df["param_a"].iloc[0]
        tmp = tmp.rename(columns={"param_value": param_name})
        return plot_parameter_sensitivity(
            tmp, param_col=param_name, metric=metric,
            algorithm=algorithm, title=title, save_path=save_path,
        )

    _style()
    pa = df["param_a"].iloc[0]
    pb = df["param_b"].iloc[0]
    pivot = (df.groupby(["value_a", "value_b"])[metric]
             .mean().unstack("value_b"))
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel(pb.replace("_", " ").title())
    ax.set_ylabel(pa.replace("_", " ").title())

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    al = f" — {algorithm}" if algorithm else ""
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} sensitivity{al}",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label=METRIC_LABELS.get(metric, metric))
    fig.tight_layout()
    _save(fig, save_path)
    return fig
