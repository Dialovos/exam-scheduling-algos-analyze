"""Convergence / scaling / time-series plots.

Anything whose horizontal axis is iterations, runtime, or dataset size ends up
here. If a figure answers "how does this algo change with something
continuous?" it lives in this module.

Includes:
    * ``plot_runtime_vs_quality`` — matplotlib scatter
    * ``plot_scaling``            — matplotlib line with error bars
    * ``plot_convergence``        — matplotlib iteration-vs-fitness overlay
    * ``plot_line_across_datasets`` — matplotlib line across datasets
    * ``plot_continuous_scan``    — dual-axis soft/runtime/memory scan
    * ``plot_algo_scatter``       — Plotly quality-vs-runtime
"""
from __future__ import annotations

import numpy as np

from utils.plots.shared import (
    ALGO_COLORS, ALGO_MARKERS, ALGO_SHORT, METRIC_LABELS, HAS_MPL,
    _c, _m, _short, _order, _style, _kfmt, _save,
    group_by_family, FAMILY_COLORS,
)

# Private helpers shared with comparative.py — import lazily to avoid a cycle.
from utils.plots.comparative import (
    _hex_to_rgba, _save_plotly, _mem_unit, _PLOTLY_MARKERS,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    pass


def _family_grid(families):
    """Pick subplot grid + figsize for *families* list.

    1 family → 1x1, 2 → 1x2, 3 → 1x3, 4 → 2x2. Keeps aspect ratios readable
    without blank panels.
    """
    n = len(families)
    if n <= 1:
        return (1, 1, (8, 5))
    if n == 2:
        return (1, 2, (14, 5.5))
    if n == 3:
        return (1, 3, (18, 5.5))
    return (2, 2, (15, 10))


def _style_family_title(ax, family):
    """Tint the subplot title to the family accent color."""
    ax.set_title(family, fontweight="bold",
                 color=FAMILY_COLORS.get(family, "#333"))


def plot_runtime_vs_quality(df, dataset=None, title=None, save_path=None,
                            aggregate=True, by_family=False):
    """Quality-vs-runtime scatter.

    With ``aggregate=True`` (default) each algorithm becomes one point at
    (mean runtime, mean soft penalty) across seeds and datasets — the only
    way the plot stays readable once you have 13 algos × 8 datasets × N
    seeds (~300 points would be confetti). Pass ``aggregate=False`` for
    the raw per-run scatter when you want to see variance directly.

    ``by_family=True`` splits the figure into one subplot per search-paradigm
    family. Each panel keeps its own axis limits so intra-family variance is
    visible — at 13 algos this reads much cleaner than one crowded canvas.
    """
    if not HAS_MPL:
        return
    _style()
    data = df[df["feasible"] == True].copy()
    if dataset:
        data = data[data["dataset"] == dataset]
    if data.empty:
        return

    algos = _order(data["algorithm"].unique())

    if by_family:
        families = group_by_family(algos)
        rows, cols, figsize = _family_grid(families)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, (family, fam_algos) in enumerate(families):
            ax = axes[i]
            _scatter_rtq(ax, data, fam_algos, aggregate)
            ax.set_xlabel("Runtime (s)")
            ax.set_ylabel("Soft Penalty")
            _kfmt(ax)
            _style_family_title(ax, family)
            ax.legend(fontsize=8, loc="best")
        for j in range(len(families), len(axes)):
            axes[j].set_visible(False)
        suffix = " — " + dataset if dataset else ""
        fig.suptitle(title or f"Quality vs Runtime — by Family{suffix}",
                     fontweight="bold", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, save_path)
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    _scatter_rtq(ax, data, algos, aggregate)
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Soft Penalty")
    _kfmt(ax)
    suffix = " — " + dataset if dataset else (" (mean across datasets & seeds)" if aggregate else "")
    ax.set_title(title or f"Quality vs Runtime{suffix}", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85, loc="best", ncol=2 if len(algos) > 8 else 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _scatter_rtq(ax, data, algos, aggregate):
    """Draw the runtime-vs-quality points on *ax* — shared by full + facet paths."""
    if aggregate:
        g = (data.groupby("algorithm")
                  .agg(rt_mean=("runtime", "mean"),
                       rt_std =("runtime", "std"),
                       soft_mean=("soft_penalty", "mean"),
                       soft_std =("soft_penalty", "std"))
                  .reindex(algos).dropna(how="all").reset_index())
        for _, r in g.iterrows():
            algo = r["algorithm"]
            ax.errorbar(r["rt_mean"], r["soft_mean"],
                        xerr=r["rt_std"], yerr=r["soft_std"],
                        fmt="none", ecolor=_c(algo), alpha=0.35,
                        elinewidth=1.1, capsize=3, zorder=2)
            ax.scatter(r["rt_mean"], r["soft_mean"], label=_short(algo),
                       color=_c(algo), marker=_m(algo), s=150, alpha=0.92,
                       edgecolors="white", linewidth=0.9, zorder=3)
            ax.annotate(_short(algo),
                        xy=(r["rt_mean"], r["soft_mean"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", alpha=0.85)
    else:
        for algo in algos:
            adf = data[data["algorithm"] == algo]
            if adf.empty:
                continue
            ax.scatter(adf["runtime"], adf["soft_penalty"], label=_short(algo),
                       color=_c(algo), marker=_m(algo), s=60, alpha=0.55,
                       edgecolors="white", linewidth=0.5, zorder=3)


def plot_scaling(df, x_col="num_exams", y_col="runtime", title=None, save_path=None,
                 top_n=None, rank_by="soft_penalty", by_family=False):
    """Line chart of ``y_col`` vs ``x_col``, one line per algorithm.

    ``top_n`` (optional): with 13 algorithms the default overlay becomes
    spaghetti. Set ``top_n=5`` to bold the 5 best algorithms (ranked by
    ``rank_by`` mean, lower = better) and fade the rest.

    ``by_family=True`` (mutually exclusive with top_n in practice): renders one
    subplot per search-paradigm family so each panel carries 1–7 lines instead
    of 13.
    """
    if not HAS_MPL:
        return
    _style()
    algos = _order(df["algorithm"].unique())

    if by_family:
        families = group_by_family(algos)
        rows, cols, figsize = _family_grid(families)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, (family, fam_algos) in enumerate(families):
            ax = axes[i]
            _draw_scaling(ax, df, fam_algos, x_col, y_col, highlighted=None)
            ax.set_xlabel(x_col.replace("_", " ").title())
            ax.set_ylabel(METRIC_LABELS.get(y_col, y_col.replace("_", " ").title()))
            _kfmt(ax)
            _style_family_title(ax, family)
            ax.legend(fontsize=8, loc="best")
        for j in range(len(families), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title or f"{METRIC_LABELS.get(y_col, y_col.title())} "
                     f"vs {x_col.replace('_', ' ').title()} — by Family",
                     fontweight="bold", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, save_path)
        return fig

    highlighted = _pick_highlighted(df, algos, top_n, rank_by)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_scaling(ax, df, algos, x_col, y_col, highlighted)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(METRIC_LABELS.get(y_col, y_col.replace("_", " ").title()))
    _kfmt(ax)
    ax.set_title(title or f"{METRIC_LABELS.get(y_col, y_col.replace('_', ' ').title())} "
                 f"vs {x_col.replace('_', ' ').title()}", fontweight="bold")
    ax.legend(fontsize=9, ncol=2 if len(algos) > 8 and highlighted is None else 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _draw_scaling(ax, df, algos, x_col, y_col, highlighted):
    """Draw per-algo error-bar lines on *ax*. Shared by full + facet paths."""
    for algo in algos:
        adf = df[df["algorithm"] == algo]
        if adf.empty:
            continue
        g = (adf.groupby(x_col)[y_col]
             .agg(["mean", "std"]).reset_index().sort_values(x_col))
        if highlighted is None or algo in highlighted:
            ax.errorbar(g[x_col], g["mean"], yerr=g["std"], label=_short(algo),
                        color=_c(algo), marker=_m(algo), linewidth=2.2,
                        markersize=8, capsize=4, alpha=0.9, zorder=3)
        else:
            ax.plot(g[x_col], g["mean"], color=_c(algo), linewidth=1.0,
                    alpha=0.22, zorder=1)


def _pick_highlighted(df, algos, top_n, rank_by):
    """Pick the top-N algorithms by ``rank_by`` mean (lower is better).

    Returns ``None`` when ``top_n`` is not set or the data doesn't justify
    filtering (n ≤ top_n, or the rank column is missing).
    """
    if top_n is None or top_n <= 0 or len(algos) <= top_n:
        return None
    if rank_by not in df.columns:
        return None
    scores = (df.groupby("algorithm")[rank_by].mean()
                .reindex(algos).dropna().sort_values())
    return set(scores.head(top_n).index.tolist())


def plot_convergence(traces, title=None, save_path=None, top_n=None,
                     by_family=False):
    """Line chart of convergence curves.

    Args:
        traces: ``{algo_name: [(iteration, fitness), ...]}``
        top_n: if set, bold the N algos with the lowest final fitness and
            fade the rest. 13 overlaid convergence curves are unreadable
            without filtering.
        by_family: split into subplots by search-paradigm family.
    """
    if not HAS_MPL or not traces:
        return
    _style()
    algos = _order([a for a in traces if traces[a]])

    if by_family:
        families = group_by_family(algos)
        rows, cols, figsize = _family_grid(families)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, (family, fam_algos) in enumerate(families):
            ax = axes[i]
            _draw_convergence(ax, traces, fam_algos, highlighted=None)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness (Hard * 100k + Soft)")
            _kfmt(ax)
            _style_family_title(ax, family)
            ax.legend(fontsize=8, loc="best")
        for j in range(len(families), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title or "Convergence — by Family",
                     fontweight="bold", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, save_path)
        return fig

    highlighted = None
    if top_n and len(algos) > top_n:
        finals = {a: traces[a][-1][1] for a in algos}
        highlighted = set(sorted(finals, key=finals.get)[:top_n])

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_convergence(ax, traces, algos, highlighted)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (Hard * 100k + Soft)")
    _kfmt(ax)
    ax.set_title(title or "Convergence Comparison", fontweight="bold")
    ax.legend(fontsize=9, ncol=2 if len(algos) > 8 and highlighted is None else 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _draw_convergence(ax, traces, algos, highlighted):
    """Draw convergence curves for *algos* on *ax*. Shared helper."""
    for algo in algos:
        data = traces.get(algo)
        if not data:
            continue
        iters, fitness = zip(*data)
        if highlighted is None or algo in highlighted:
            ax.plot(iters, fitness, label=_short(algo), color=_c(algo),
                    linewidth=2.2, alpha=0.9, zorder=3)
        else:
            ax.plot(iters, fitness, color=_c(algo),
                    linewidth=1.0, alpha=0.22, zorder=1)


def plot_line_across_datasets(df, metric="soft_penalty", title=None, save_path=None,
                              top_n=None, by_family=False):
    """Line chart — one line per algorithm, x = dataset, y = metric mean.

    ``top_n`` (optional): bold the top-N algorithms by ``metric`` mean and
    fade the rest.

    ``by_family=True``: split into subplots by search-paradigm family. Each
    panel inherits the shared x-axis of dataset names.
    """
    if not HAS_MPL:
        return
    _style()
    datasets = list(df["dataset"].unique())
    if len(datasets) < 2:
        return
    x = np.arange(len(datasets))
    algos = _order(df["algorithm"].unique())

    if by_family:
        families = group_by_family(algos)
        rows, cols, figsize = _family_grid(families)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, (family, fam_algos) in enumerate(families):
            ax = axes[i]
            _draw_lines_across(ax, df, fam_algos, datasets, metric, highlighted=None)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, fontsize=8)
            if len(datasets) >= 4:
                ax.tick_params(axis="x", rotation=25)
                for lbl in ax.get_xticklabels():
                    lbl.set_ha("right")
            _kfmt(ax)
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            _style_family_title(ax, family)
            ax.legend(fontsize=8, loc="best")
        for j in range(len(families), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title or f"{METRIC_LABELS.get(metric, metric)} Across Datasets — by Family",
                     fontweight="bold", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, save_path)
        return fig

    highlighted = _pick_highlighted(df, algos, top_n, metric)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 6))
    _draw_lines_across(ax, df, algos, datasets, metric, highlighted)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    if len(datasets) >= 4:
        ax.tick_params(axis="x", rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    _kfmt(ax)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} Across Datasets",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2 if len(algos) > 8 and highlighted is None else 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _draw_lines_across(ax, df, algos, datasets, metric, highlighted):
    """Plot one errorbar line per algo across *datasets*. Shared helper."""
    x = np.arange(len(datasets))
    for algo in algos:
        adf = df[df["algorithm"] == algo]
        if adf.empty:
            continue
        means = [adf[adf["dataset"] == ds][metric].mean() for ds in datasets]
        stds = [adf[adf["dataset"] == ds][metric].std() for ds in datasets]
        stds = [0 if np.isnan(s) else s for s in stds]
        if highlighted is None or algo in highlighted:
            ax.errorbar(x, means, yerr=stds, label=_short(algo),
                        color=_c(algo), marker=_m(algo), linewidth=2.2,
                        markersize=8, capsize=4, alpha=0.9, zorder=3)
        else:
            ax.plot(x, means, color=_c(algo), linewidth=1.0,
                    alpha=0.22, zorder=1)


def plot_continuous_scan(df, x_col="num_exams", title=None, save_path=None):
    """Continuous-axis line chart for size/parameter scans.

    Two subplots side-by-side: soft penalty on the left, runtime + memory
    (dual-axis) on the right. Aggregates rows by ``x_col`` using mean + std.
    """
    if not HAS_MPL:
        return
    _style()

    required = {x_col, "soft_penalty", "runtime", "memory_peak_mb"}
    missing = required - set(df.columns)
    if missing:
        print(f"[plot_continuous_scan] missing columns: {missing}")
        return

    grouped = df.groupby(x_col).agg(
        soft_mean=("soft_penalty", "mean"),
        soft_std =("soft_penalty", "std"),
        rt_mean  =("runtime", "mean"),
        rt_std   =("runtime", "std"),
        mem_mean =("memory_peak_mb", "mean"),
        mem_std  =("memory_peak_mb", "std"),
    ).reset_index().fillna(0)

    x = grouped[x_col].values

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    soft_color = "#E15759"
    ax_l.plot(x, grouped["soft_mean"], "-o", color=soft_color,
              linewidth=2.2, markersize=7, label="Soft Penalty")
    ax_l.fill_between(x,
                      grouped["soft_mean"] - grouped["soft_std"],
                      grouped["soft_mean"] + grouped["soft_std"],
                      color=soft_color, alpha=0.2)
    ax_l.set_xlabel(x_col.replace("_", " ").title())
    ax_l.set_ylabel(METRIC_LABELS.get("soft_penalty", "Soft Penalty"))
    ax_l.set_title("Quality", fontweight="bold")
    ax_l.grid(True, alpha=0.3)
    _kfmt(ax_l)

    rt_color  = "#F28E2B"
    mem_color = "#4E79A7"

    l1 = ax_r.plot(x, grouped["rt_mean"], "-o", color=rt_color,
                   linewidth=2.2, markersize=7, label="Runtime (s)")
    ax_r.fill_between(x,
                      grouped["rt_mean"] - grouped["rt_std"],
                      grouped["rt_mean"] + grouped["rt_std"],
                      color=rt_color, alpha=0.2)
    ax_r.set_xlabel(x_col.replace("_", " ").title())
    ax_r.set_ylabel("Runtime (s)", color=rt_color)
    ax_r.tick_params(axis="y", labelcolor=rt_color)
    ax_r.grid(True, alpha=0.3)

    ax_r2 = ax_r.twinx()
    mscale, munit, _fmt = _mem_unit(grouped["mem_mean"].tolist())
    mem_scaled = grouped["mem_mean"] * mscale
    mem_std_scaled = grouped["mem_std"] * mscale
    l2 = ax_r2.plot(x, mem_scaled, "-s", color=mem_color,
                    linewidth=2.2, markersize=7, label=f"Peak Memory ({munit})")
    ax_r2.fill_between(x,
                       mem_scaled - mem_std_scaled,
                       mem_scaled + mem_std_scaled,
                       color=mem_color, alpha=0.2)
    ax_r2.set_ylabel(f"Peak Memory ({munit})", color=mem_color)
    ax_r2.tick_params(axis="y", labelcolor=mem_color)
    ax_r2.grid(False)

    ax_r.set_title("Cost", fontweight="bold")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax_r.legend(lines, labels, loc="upper left", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Plotly scatter ────────────────────────────────────────────────────────
def plot_algo_scatter(df, save_path=None):
    """Scatter: quality (y) vs runtime (x), one point per algorithm."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot_algo_scatter] pip install plotly"); return None

    feas = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_scatter] No feasible runs"); return None

    g = feas.groupby("algorithm").agg(
        soft_m=("soft_penalty", "mean"), rt_m=("runtime", "mean"),
    ).reset_index()

    algos = g["algorithm"].tolist()
    shorts = [ALGO_SHORT.get(a, a) for a in algos]
    colors = [ALGO_COLORS.get(a, "#888888") for a in algos]
    syms = [_PLOTLY_MARKERS.get(ALGO_MARKERS.get(a, "o"), "circle") for a in algos]

    fig = go.Figure()
    for i, algo in enumerate(algos):
        r = g[g["algorithm"] == algo].iloc[0]
        fig.add_trace(go.Scatter(
            x=[r["rt_m"]], y=[r["soft_m"]],
            mode="markers",
            marker=dict(size=13, color=colors[i], symbol=syms[i],
                        line=dict(width=1.2, color="white")),
            name=shorts[i], showlegend=True,
            hovertemplate=(f"<b>{shorts[i]}</b><br>Soft: %{{y:,.0f}}<br>"
                           f"Time: %{{x:.2f}}s<extra></extra>"),
        ))

    fig.update_layout(
        width=700, height=520,
        template="plotly_white",
        title=dict(text="<b>Quality vs Runtime</b>", x=0.5, font_size=14),
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12),
        margin=dict(t=55, b=50, l=70, r=140),
        xaxis=dict(title="Mean Runtime (s)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="Mean Soft Penalty", gridcolor="rgba(0,0,0,0.06)"),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.02,
                    font_size=10),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig
