"""Comparative plots — one figure answers one question about how algos stack up.

Covers the "horizontal bar / box / radar / heatmap / Pareto" family of charts
we use to place algorithms next to each other on a single dataset (or pooled
across datasets). Matplotlib figures use the paper palette from
:mod:`utils.plots.shared`; Plotly figures reuse the same palette so print and
interactive notebook figures look identical.
"""
from __future__ import annotations

from utils.plots.shared import (
    ALGO_COLORS, ALGO_MARKERS, ALGO_SHORT, ALGO_ORDER, SOFT_KEYS, SOFT_LABELS,
    SOFT_COLORS, METRIC_LABELS, HAS_MPL,
    _c, _m, _short, _order, _style, _kfmt, _apply_xlabels, _save,
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover
    pass


# ── Matplotlib: horizontal bars ──────────────────────────────────────────
def plot_algorithm_comparison(df, dataset=None, metric="soft_penalty",
                              title=None, save_path=None, ax=None):
    """Horizontal bar chart — values on the right, names on the left.
    Eliminates label overlap entirely."""
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df.copy()
    grouped = (data.groupby("algorithm")[metric]
               .agg(["mean", "std", "min", "count"])
               .reindex(_order(data["algorithm"].unique()))
               .dropna(subset=["mean"]))
    if grouped.empty:
        return

    own_fig = ax is None
    n = len(grouped)
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, max(3, n * 0.65)))
    else:
        fig = ax.figure

    y = np.arange(n)
    colors = [_c(a) for a in grouped.index]
    ax.barh(y, grouped["mean"], xerr=grouped["std"],
            color=colors, alpha=0.88, height=0.6,
            capsize=3, edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1})

    x_max = (grouped["mean"] + grouped["std"]).max()
    for i, (_, row) in enumerate(grouped.iterrows()):
        lbl = f"{row['mean']:,.0f}" if row["mean"] >= 10 else f"{row['mean']:.2f}"
        if row["count"] > 1:
            lbl += f"  (n={int(row['count'])})"
        ax.text(row["mean"] + row["std"] + x_max * 0.02, i,
                lbl, va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in grouped.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.30)
    _kfmt(ax, "x")
    mlabel = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(mlabel)
    ds = f" — {dataset}" if dataset else ""
    ax.set_title(title or f"{mlabel}{ds}", fontweight="bold")

    if own_fig:
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_multi_dataset_heatmap(df, metric="soft_penalty", title=None, save_path=None):
    if not HAS_MPL:
        return
    _style()
    pivot = df.pivot_table(values=metric, index="algorithm", columns="dataset", aggfunc="mean")
    if pivot.empty:
        return
    ordered = [a for a in ALGO_ORDER if a in pivot.index]
    leftover = [a for a in pivot.index if a not in ordered]
    pivot = pivot.reindex(ordered + leftover)

    nr, nc = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, nc * 2.2), max(3, nr * 0.7)))
    im = ax.imshow(pivot.values, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(nc))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([_short(a) for a in pivot.index], fontsize=10)

    thresh = np.nanmean(pivot.values)
    for i in range(nr):
        for j in range(nc):
            v = pivot.values[i, j]
            if not np.isnan(v):
                c = "white" if v > thresh else "black"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=9, color=c, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8,
                 label=METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Cross-Dataset: {METRIC_LABELS.get(metric, metric)}",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_feasibility_rates(df, title=None, save_path=None):
    if not HAS_MPL:
        return
    _style()
    ordered = _order(df["algorithm"].unique())
    rates = df.groupby("algorithm")["feasible"].mean().reindex(ordered).dropna() * 100
    n = len(rates)

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.55)))
    y = np.arange(n)
    ax.barh(y, rates.values, color=[_c(a) for a in rates.index],
            alpha=0.88, height=0.55, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(rates.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in rates.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("Feasibility Rate (%)")
    ax.set_title(title or "Feasibility Rate by Algorithm", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_box_comparison(df, dataset=None, metric="soft_penalty",
                        title=None, save_path=None):
    """Horizontal box + jittered strip — readable with 10+ algorithms.

    Horizontal because labels never clip or rotate; jittered strip overlay
    because the box alone hides sample size; sorted by median so the best
    algorithm sits on top after the y-axis inverts.
    """
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df.copy()
    algos = _order(data["algorithm"].unique())
    box_data, valid_algos = [], []
    for a in algos:
        vals = data[data["algorithm"] == a][metric].dropna().values
        if len(vals) > 0:
            box_data.append(vals)
            valid_algos.append(a)
    if not valid_algos:
        return

    medians = [np.median(d) for d in box_data]
    order = np.argsort(medians)[::-1]
    box_data = [box_data[i] for i in order]
    valid_algos = [valid_algos[i] for i in order]

    n = len(valid_algos)
    fig, ax = plt.subplots(figsize=(10, max(3.5, n * 0.6 + 1)))
    positions = list(range(n))

    bp = ax.boxplot(
        box_data, positions=positions, vert=False, patch_artist=True,
        widths=0.45, showfliers=False,
        medianprops=dict(color="#222222", linewidth=2),
        whiskerprops=dict(color="#666666", linewidth=1, linestyle="--"),
        capprops=dict(color="#666666", linewidth=1),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#E03030",
                       markeredgecolor="white", markersize=5.5,
                       markeredgewidth=0.8, zorder=5),
    )
    for patch, algo in zip(bp["boxes"], valid_algos):
        color = _c(algo)
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    rng = np.random.RandomState(0)
    for i, (vals, algo) in enumerate(zip(box_data, valid_algos)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, i + jitter, color=_c(algo), alpha=0.65,
                   s=22, edgecolor="white", linewidth=0.4, zorder=4)

    x_max = max(np.max(d) for d in box_data)
    x_min = min(np.min(d) for d in box_data)
    pad = (x_max - x_min) * 0.02
    for i, vals in enumerate(box_data):
        med = np.median(vals)
        ax.text(x_max + pad, i, f"{med:,.0f}", va="center", ha="left",
                fontsize=8.5, color="#444444", fontstyle="italic")

    ax.set_yticks(positions)
    ax.set_yticklabels([_short(a) for a in valid_algos], fontsize=10)
    ax.set_xlim(x_min - (x_max - x_min) * 0.05,
                x_max + (x_max - x_min) * 0.18)
    _kfmt(ax, "x")
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E03030",
               markeredgecolor="white", markersize=5.5, label="Mean"),
        Line2D([0], [0], color="#222222", linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.8)

    ax.set_title(title or f"Distribution: {METRIC_LABELS.get(metric, metric)}"
                 f"{' — ' + dataset if dataset else ''}", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_radar(df, dataset=None, title=None, save_path=None):
    """Radar chart — each axis is a soft component normalized to [0, 1].
    Smaller polygon = better."""
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df.copy()
    means = data.groupby("algorithm")[SOFT_KEYS].mean()
    algos = [a for a in _order(means.index) if a in means.index]
    if len(algos) < 2:
        return

    vals = means.loc[algos].values
    col_max = vals.max(axis=0)
    col_max[col_max == 0] = 1
    norm = vals / col_max

    n_cats = len(SOFT_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    for i, algo in enumerate(algos):
        values = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, values, "o-", linewidth=2, label=_short(algo),
                color=_c(algo), markersize=5, alpha=0.85)
        ax.fill(angles, values, alpha=0.06, color=_c(algo))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SOFT_LABELS, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title(title or f"Performance Profile"
                 f"{' — ' + dataset if dataset else ''}",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_rank_table(df, dataset=None, title=None, save_path=None):
    """Color-coded ranking table: soft penalty, feasibility, runtime, rank."""
    if not HAS_MPL:
        return
    _style()
    data = df[df["dataset"] == dataset] if dataset else df.copy()
    if data.empty:
        return

    stats = data.groupby("algorithm").agg(
        soft_mean=("soft_penalty", "mean"),
        soft_std=("soft_penalty", "std"),
        feasible_pct=("feasible", lambda x: x.mean() * 100),
        runtime_mean=("runtime", "mean"),
        n_runs=("soft_penalty", "count"),
    ).fillna(0).sort_values("soft_mean")
    stats["rank"] = range(1, len(stats) + 1)

    algos = stats.index.tolist()
    col_labels = ["Rank", "Algorithm", "Soft Penalty", "Std Dev",
                  "Feasible %", "Runtime (s)", "Runs"]

    cell_text = []
    for algo in algos:
        r = stats.loc[algo]
        cell_text.append([
            f"{int(r['rank'])}", _short(algo),
            f"{r['soft_mean']:,.0f}", f"{r['soft_std']:,.0f}",
            f"{r['feasible_pct']:.0f}%", f"{r['runtime_mean']:.2f}",
            f"{int(r['n_runs'])}",
        ])

    n = len(algos)
    fig, ax = plt.subplots(figsize=(11, max(2.5, n * 0.5 + 1.5)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(n):
        frac = i / max(n - 1, 1)
        rgb = (int(50 + frac * 180), int(200 - frac * 130), int(80 - frac * 30))
        bg = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_alpha(0.25)

    ax.set_title(title or f"Algorithm Ranking"
                 f"{' — ' + dataset if dataset else ''}",
                 fontweight="bold", fontsize=13, pad=20)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Plotly: interactive summaries ────────────────────────────────────────
def _hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


_PLOTLY_MARKERS = {
    "o": "circle", "^": "triangle-up", "v": "triangle-down",
    "s": "square", "P": "cross", "X": "x", "h": "hexagon",
    "p": "pentagon", "H": "hexagon2", "D": "diamond",
    "d": "diamond-wide", "*": "star",
}


def _save_plotly(fig, save_path):
    """Save a Plotly figure — HTML (no deps) or image (needs kaleido)."""
    if save_path.endswith(".html"):
        fig.write_html(save_path, include_plotlyjs="cdn")
    else:
        fig.write_image(save_path)


def _mem_unit(mb_values):
    """Pick best memory unit based on max value. Returns (scale, label, fmt)."""
    mx = max(mb_values) if mb_values else 0
    if mx >= 1.0:
        return 1.0, "MB", ".1f"
    if mx >= 0.001:
        return 1024.0, "KB", ".0f"
    return 1024.0 * 1024.0, "B", ".0f"


def plot_experiment_summary(df, save_path=None):
    """Interactive summary: soft penalty, runtime, and memory across datasets.

    Three-panel line chart with a categorical (evenly-spaced) x-axis so
    datasets with similar sizes don't overlap. One line per algorithm with
    ±1σ ribbon. Requires ``pip install plotly``.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as _np
    except ImportError:
        print("[plot_experiment_summary] pip install plotly")
        return None

    feasible = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feasible.empty:
        print("[plot_experiment_summary] No feasible runs to plot")
        return None

    has_mem = "memory_peak_mb" in feasible.columns
    agg_dict = {
        "soft_m": ("soft_penalty", "mean"), "soft_s": ("soft_penalty", "std"),
        "rt_m":   ("runtime", "mean"),      "rt_s":   ("runtime", "std"),
    }
    if has_mem:
        agg_dict["mem_m"] = ("memory_peak_mb", "mean")
        agg_dict["mem_s"] = ("memory_peak_mb", "std")

    ds_size = (feasible.groupby("dataset")["num_exams"].first()
                       .sort_values().reset_index())
    ds_size["label"] = ds_size.apply(
        lambda r: f"{r['dataset'].replace('exam_comp_', '')} ({r['num_exams']})", axis=1)
    label_order = ds_size["label"].tolist()
    ds_to_label = dict(zip(ds_size["dataset"], ds_size["label"]))

    feasible["ds_label"] = feasible["dataset"].map(ds_to_label)

    g = feasible.groupby(["ds_label", "algorithm"]).agg(**agg_dict).reset_index().fillna(0)

    if has_mem:
        mscale, munit, mfmt = _mem_unit(g["mem_m"].tolist())
        g["mem_m"] = g["mem_m"] * mscale
        g["mem_s"] = g["mem_s"] * mscale

    algos = _order([a for a in g["algorithm"].unique()])

    ncols = 3 if has_mem else 2
    titles = ["<b>Soft Penalty</b>", "<b>Runtime</b>"]
    if has_mem:
        titles.append("<b>Peak Memory</b>")

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.07)

    panels = [
        (1, "soft_m", "soft_s", "Soft", "%{y:,.0f}"),
        (2, "rt_m",   "rt_s",   "Time", "%{y:.2f}s"),
    ]
    if has_mem:
        panels.append((3, "mem_m", "mem_s", "Mem", f"%{{y:{mfmt}}} {munit}"))

    for algo in algos:
        ad = g[g["algorithm"] == algo].copy()
        if ad.empty:
            continue
        ad["_ord"] = ad["ds_label"].map({l: i for i, l in enumerate(label_order)})
        ad = ad.sort_values("_ord")

        x = ad["ds_label"].values
        c = ALGO_COLORS.get(algo, "#888888")
        band = _hex_to_rgba(c, 0.10)
        short = ALGO_SHORT.get(algo, algo)
        sym = _PLOTLY_MARKERS.get(ALGO_MARKERS.get(algo, "o"), "circle")

        for col, ym, ys, _, fmt in panels:
            fig.add_trace(go.Scatter(
                x=x, y=ad[ym].values,
                name=short, legendgroup=algo,
                mode="lines+markers", showlegend=(col == 1),
                line=dict(color=c, width=2.4, shape="spline"),
                marker=dict(size=7, symbol=sym),
                hovertemplate=f"{short}: {fmt}<extra></extra>",
            ), row=1, col=col)

            upper = (ad[ym] + ad[ys]).values
            lower = _np.maximum(0, (ad[ym] - ad[ys]).values)
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=_np.concatenate([upper, lower[::-1]]),
                fill="toself", fillcolor=band, line=dict(width=0),
                showlegend=False, legendgroup=algo, hoverinfo="skip",
            ), row=1, col=col)

    fig.update_layout(
        width=1200, height=480, template="plotly_white",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=11), tracegroupgap=5,
        ),
        margin=dict(t=50, b=110, l=60, r=30),
        hovermode="x unified",
    )

    for col in range(1, ncols + 1):
        fig.update_xaxes(
            categoryorder="array", categoryarray=label_order,
            tickangle=-35, row=1, col=col,
            gridcolor="rgba(0,0,0,0.06)",
        )
    fig.update_yaxes(title_text="Soft Penalty", row=1, col=1,
                     gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(title_text="Seconds", row=1, col=2,
                     gridcolor="rgba(0,0,0,0.06)")
    if has_mem:
        fig.update_yaxes(title_text=munit, row=1, col=3,
                         gridcolor="rgba(0,0,0,0.06)")

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_bars(df, save_path=None):
    """Horizontal bar chart: soft penalty, runtime, and memory per algorithm."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[plot_algo_bars] pip install plotly"); return None

    feas = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_bars] No feasible runs"); return None

    has_mem = "memory_peak_mb" in feas.columns
    agg_dict = {
        "soft_m": ("soft_penalty", "mean"), "soft_s": ("soft_penalty", "std"),
        "rt_m":   ("runtime", "mean"),      "rt_s":   ("runtime", "std"),
        "n":      ("soft_penalty", "count"),
    }
    if has_mem:
        agg_dict["mem_m"] = ("memory_peak_mb", "mean")
        agg_dict["mem_s"] = ("memory_peak_mb", "std")

    g = feas.groupby("algorithm").agg(**agg_dict).reset_index().fillna(0)

    if has_mem:
        mscale, munit, mfmt = _mem_unit(g["mem_m"].tolist())
        g["mem_m"] = g["mem_m"] * mscale
        g["mem_s"] = g["mem_s"] * mscale

    ncols = 3 if has_mem else 2
    titles = ["<b>Soft Penalty</b>", "<b>Runtime</b>"]
    if has_mem:
        titles.append("<b>Peak Memory</b>")

    panels = [
        ("soft_m", "soft_s", "Soft Penalty", "%{x:,.0f}"),
        ("rt_m",   "rt_s",   "Seconds",      "%{x:.2f}s"),
    ]
    if has_mem:
        panels.append(("mem_m", "mem_s", munit, f"%{{x:{mfmt}}} {munit}"))

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.14)

    for col, (ym, ys, xlabel, fmt) in enumerate(panels, 1):
        gs = g.sort_values(ym, ascending=True)
        shorts = [ALGO_SHORT.get(a, a) for a in gs["algorithm"]]
        colors = [ALGO_COLORS.get(a, "#888888") for a in gs["algorithm"]]
        fig.add_trace(go.Bar(
            y=shorts, x=gs[ym].values, orientation="h",
            error_x=dict(type="data", array=gs[ys].values, thickness=1.2),
            marker_color=colors, showlegend=False,
            hovertemplate=f"%{{y}}: {fmt}<extra></extra>",
        ), row=1, col=col)
        fig.update_xaxes(title_text=xlabel, row=1, col=col,
                         gridcolor="rgba(0,0,0,0.06)")

    h = max(350, len(g) * 38)
    fig.update_layout(
        width=1150 if has_mem else 1050, height=h,
        template="plotly_white",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12),
        margin=dict(t=45, b=30, l=80, r=30),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_boxes(df, save_path=None):
    """Box plot: soft penalty distribution per algorithm, all datasets pooled."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot_algo_boxes] pip install plotly"); return None

    feas = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_boxes] No feasible runs"); return None

    medians = feas.groupby("algorithm")["soft_penalty"].median().sort_values()
    algos = medians.index.tolist()

    fig = go.Figure()
    for algo in algos:
        ad = feas[feas["algorithm"] == algo]
        short = ALGO_SHORT.get(algo, algo)
        fig.add_trace(go.Box(
            y=ad["soft_penalty"].values, name=short,
            marker_color=ALGO_COLORS.get(algo, "#888888"),
            boxpoints="all", jitter=0.4, pointpos=-1.5,
            marker=dict(size=4, opacity=0.5),
            line=dict(width=1.8),
            hoverinfo="y+name",
        ))

    fig.update_layout(
        width=1050, height=480, template="plotly_white",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12),
        yaxis_title="Soft Penalty", showlegend=False,
        title=dict(text="<b>Soft Penalty Distribution</b>", x=0.5, font_size=14),
        margin=dict(t=50, b=40, l=65, r=30),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_radar(df, save_path=None):
    """Radar chart: normalised performance profile per algorithm (smaller = better)."""
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError:
        print("[plot_algo_radar] pip install plotly"); return None

    feas = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_radar] No feasible runs"); return None

    metrics = [("soft_penalty", "Soft Penalty", False),
               ("runtime", "Runtime", False),
               ("memory_peak_mb", "Memory", False),
               ("two_in_a_row", "2-in-a-Row", False),
               ("two_in_a_day", "2-in-a-Day", False),
               ("period_spread", "Period Spread", False),
               ("non_mixed_durations", "Mixed Dur.", False),
               ("front_load", "Front Load", False)]
    metrics = [(k, l, inv) for k, l, inv in metrics if k in feas.columns]
    if len(metrics) < 3:
        print("[plot_algo_radar] Need >= 3 metrics"); return None

    g = feas.groupby("algorithm")[[m[0] for m in metrics]].mean()
    algos = _order([a for a in g.index])

    normed = g.copy()
    for col, _, inv in metrics:
        mn, mx = g[col].min(), g[col].max()
        if mx > mn:
            normed[col] = (g[col] - mn) / (mx - mn)
            if inv:
                normed[col] = 1.0 - normed[col]
        else:
            normed[col] = 0.0

    cats = [m[1] for m in metrics]

    fig = go.Figure()
    for algo in algos:
        if algo not in normed.index:
            continue
        vals = [normed.loc[algo, m[0]] for m in metrics]
        vals.append(vals[0])
        short = ALGO_SHORT.get(algo, algo)
        c = ALGO_COLORS.get(algo, "#888888")
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            name=short, line=dict(color=c, width=2.2),
            fill="toself", fillcolor=_hex_to_rgba(c, 0.08),
        ))

    fig.update_layout(
        width=650, height=520, template="plotly_white",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=11),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05],
                            gridcolor="rgba(0,0,0,0.08)", tickfont_size=9),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
        ),
        title=dict(text="<b>Performance Profile</b> <sub>(smaller = better)</sub>",
                   x=0.5, font_size=14),
        legend=dict(font_size=10),
        margin=dict(t=70, b=40, l=65, r=65),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_heatmap(df, save_path=None):
    """Heatmap: algorithm x dataset, coloured by mean soft penalty.

    Cells with no feasible result show 'n/f' (not feasible) in grey. Init-only
    algorithms (Greedy, Feasibility, Seeder) are excluded — they don't run on
    every set, and including them makes the grid look like a fence.
    """
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError:
        print("[plot_algo_heatmap] pip install plotly"); return None

    feas = df[df["feasible"] == True].copy() if "feasible" in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_heatmap] No feasible runs"); return None

    skip = {"Greedy", "Feasibility", "Seeder"}
    feas = feas[~feas["algorithm"].isin(skip)]
    if feas.empty:
        print("[plot_algo_heatmap] No search-algorithm data"); return None

    all_algos = sorted(df[~df["algorithm"].isin(skip)]["algorithm"].unique())
    all_datasets = sorted(df["dataset"].unique())

    g = feas.groupby(["algorithm", "dataset"])["soft_penalty"].mean().reset_index()
    pivot = g.pivot(index="algorithm", columns="dataset", values="soft_penalty")
    pivot = pivot.reindex(index=all_algos, columns=all_datasets)

    infeas_set = set()
    for algo in all_algos:
        for ds in all_datasets:
            sub = df[(df["algorithm"] == algo) & (df["dataset"] == ds)]
            if len(sub) > 0 and sub["feasible"].sum() == 0:
                infeas_set.add((algo, ds))

    ds_size = df.groupby("dataset")["num_exams"].first().sort_values()
    ds_order = [d for d in ds_size.index if d in pivot.columns]
    ds_labels = [d.replace("exam_comp_", "") for d in ds_order]

    algo_means = pivot[ds_order].mean(axis=1).sort_values()
    algo_order = [a for a in algo_means.index if a in pivot.index]
    algo_labels = [ALGO_SHORT.get(a, a) for a in algo_order]

    z = pivot.loc[algo_order, ds_order].values

    z_norm = _np.full_like(z, dtype=float, fill_value=_np.nan)
    for j in range(z.shape[1]):
        col = z[:, j]
        valid = ~_np.isnan(col)
        if valid.any():
            mn, mx = _np.nanmin(col), _np.nanmax(col)
            if mx > mn:
                z_norm[:, j] = (col - mn) / (mx - mn)
            else:
                z_norm[valid, j] = 0.0

    text = []
    for i, algo in enumerate(algo_order):
        row_text = []
        for j, ds in enumerate(ds_order):
            v = z[i, j]
            if not _np.isnan(v):
                row_text.append(f"{v:,.0f}")
            elif (algo, ds) in infeas_set:
                row_text.append("n/f")
            else:
                row_text.append("-")
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z_norm, x=ds_labels, y=algo_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale="RdYlGn_r", showscale=True,
        colorbar=dict(title="Relative", tickvals=[0, 0.5, 1],
                      ticktext=["Best", "Mid", "Worst"]),
        hovertemplate=("<b>%{y}</b> on %{x}<br>"
                       "Soft: %{text}<extra></extra>"),
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        width=max(550, len(ds_order) * 85 + 180),
        height=max(400, len(algo_order) * 38 + 120),
        template="plotly_white",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12),
        title=dict(text="<b>Soft Penalty Heatmap</b> <sub>(per-dataset normalised, n/f = not feasible)</sub>",
                   x=0.5, font_size=14),
        xaxis=dict(title="", side="bottom"),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(t=65, b=40, l=80, r=30),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig
