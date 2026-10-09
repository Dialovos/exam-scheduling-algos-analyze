"""README figures from the cached paper batch, with light and dark themes."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from utils.plots.shared import algo_family, algo_short, load_batch018, normalize_per_instance
from utils.plots.paper.fig1_pareto import _pareto_front
from utils.plots.paper.fig5_sensitivity import ALGO_CANONICAL, _canonical_param, _sensitivity
from utils.plots.paper.fig8_gap_leaderboard import EXCLUDE_INSTANCES


THEMES = {
    "light": dict(background="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", baseline="#c3c2b7"),
    "dark": dict(background="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", baseline="#383835"),
}
FAMILY_COLORS = {
    "light": {"Trajectory": "#2a78d6", "Population": "#eb6834",
              "Exact / Hybrid": "#1baf7a", "Construction": "#eda100"},
    "dark": {"Trajectory": "#3987e5", "Population": "#d95926",
             "Exact / Hybrid": "#199e70", "Construction": "#c98500"},
}
BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)


def readme_style(theme):
    """Return isolated Matplotlib settings for a README theme."""
    colors = THEMES[theme]
    return {
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": colors["background"], "axes.facecolor": colors["background"],
        "text.color": colors["primary"], "axes.labelcolor": colors["secondary"],
        "xtick.color": colors["muted"], "ytick.color": colors["secondary"],
        "axes.edgecolor": colors["baseline"], "axes.linewidth": 0.6,
        "axes.grid": False, "grid.color": colors["grid"],
        "grid.linestyle": "-", "grid.linewidth": 0.4, "grid.alpha": 1.0,
        "axes.axisbelow": True, "xtick.labelsize": 9, "ytick.labelsize": 10,
        "savefig.transparent": False, "savefig.bbox": None,
    }


def _canvas(theme, title, subtitle, *, height=5.9):
    fig, ax = plt.subplots(figsize=(8, height), dpi=200)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.12, top=0.80)
    fig.text(0.035, 0.96, title, ha="left", va="top", fontsize=15, fontweight="normal")
    fig.text(0.035, 0.915, subtitle, ha="left", va="top", fontsize=9,
             color=THEMES[theme]["secondary"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    return fig, ax


def _family_legend(fig, algorithms, theme):
    families = {algo_family(a) for a in algorithms}
    handles = [Patch(facecolor=color, label=family)
               for family, color in FAMILY_COLORS[theme].items() if family in families]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.865),
               ncol=len(handles), frameon=False, fontsize=8.5, handlelength=1,
               handleheight=0.8, columnspacing=1.5, labelcolor=THEMES[theme]["secondary"])


def _save(fig, out_path, theme):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=THEMES[theme]["background"], transparent=False)
    plt.close(fig)
    return path


def _label(algorithm):
    return "CP-SAT (60s)" if algo_short(algorithm) == "CP-SAT" else algo_short(algorithm)


def make_leaderboard(out_path, theme="light", *, batch=None):
    """Plot mean percent gaps to the same proved optima used by paper figure 8."""
    b = load_batch018() if batch is None else batch
    optima = {ds: sum(parts.values()) for ds, parts in b.ip_soft.items()
              if ds not in EXCLUDE_INSTANCES}
    means = (b.main[b.main["dataset"].isin(optima)]
             .groupby(["algorithm", "dataset"])["soft_penalty"].mean().unstack())
    gaps = means.copy()
    for ds, optimum in optima.items():
        gaps[ds] = (means[ds] / optimum - 1) * 100
    values = gaps.mean(axis=1).sort_values()

    with plt.rc_context(readme_style(theme)):
        fig, ax = _canvas(theme, "Gap to the proved optimum",
                          "Mean over the four ITC 2007 sets CP-SAT solved; lower is better")
        y = np.arange(len(values))
        ax.barh(y, values, left=0, height=0.7,
                color=[FAMILY_COLORS[theme][algo_family(a)] for a in values.index])
        ax.set_yticks(y, [_label(a) for a in values.index])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, values.max() * 1.65)
        ax.set_xticks([1, 10, 100, 1000], ["1%", "10%", "100%", "1000%"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Mean gap (%)", labelpad=10)
        ax.grid(axis="x")
        for i, value in enumerate(values):
            ax.annotate(f"{value:,.1f}%", (value, i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=THEMES[theme]["secondary"])
        _family_legend(fig, values.index, theme)
        return _save(fig, out_path, theme)


def _cell_ink(color, theme):
    # Choose the theme text color with the higher WCAG contrast against this cell.
    def luminance(hex_color):
        rgb = np.array(to_rgb(hex_color))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(linear @ [0.2126, 0.7152, 0.0722])

    lum = luminance(color)
    candidates = (THEMES[theme]["primary"], THEMES["dark" if theme == "light" else "light"]["primary"])
    return max(candidates, key=lambda ink: (max(lum, luminance(ink)) + 0.05)
               / (min(lum, luminance(ink)) + 0.05))


def make_per_set_heatmap(out_path, theme="light", *, batch=None):
    """Average seeds, then normalize each set to its best algorithm mean (1.00)."""
    b = load_batch018() if batch is None else batch
    means = b.main.groupby(["algorithm", "dataset"])["soft_penalty"].mean().reset_index()
    matrix = normalize_per_instance(means).pivot(index="algorithm", columns="dataset", values="soft_norm")
    order = matrix.rank(axis=0, method="min").mean(axis=1).sort_values(kind="stable").index
    matrix = matrix.loc[order]
    values = matrix.to_numpy()
    ramp = BLUE_RAMP if theme == "light" else BLUE_RAMP[::-1]
    cmap = ListedColormap(ramp)
    norm = BoundaryNorm(np.linspace(1, 2, len(ramp) + 1), len(ramp), clip=True)

    with plt.rc_context(readme_style(theme)):
        fig, ax = _canvas(theme, "Quality across all eight sets",
                          "Mean soft penalty relative to the best on each set; rows ordered by mean rank",
                          height=6.4)
        fig.subplots_adjust(top=0.78, bottom=0.19)
        mesh = ax.pcolormesh(np.minimum(values, 2), cmap=cmap, norm=norm,
                             edgecolors=THEMES[theme]["background"], linewidth=0.72,
                             antialiased=False)
        ax.set_xlim(0, len(matrix.columns))
        ax.set_ylim(len(matrix), 0)
        ax.set_xticks(np.arange(len(matrix.columns)) + 0.5,
                      [ds.replace("exam_comp_", "") for ds in matrix.columns])
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(matrix)) + 0.5, [algo_short(a) for a in order])
        ax.tick_params(axis="y", pad=15)
        ax.scatter(np.full(len(order), -0.13), np.arange(len(order)) + 0.5,
                   c=[FAMILY_COLORS[theme][algo_family(a)] for a in order], s=22,
                   clip_on=False)
        ax.spines["bottom"].set_visible(False)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j + 0.5, i + 0.5, "≥2" if value > 2 else f"{value:.2f}",
                        ha="center", va="center", fontsize=9.5,
                        fontweight="bold" if np.isclose(value, values[:, j].min()) else "normal",
                        color=_cell_ink(cmap(norm(value)), theme))
        scale_ax = fig.add_axes([0.25, 0.105, 0.55, 0.018])
        cbar = fig.colorbar(mesh, cax=scale_ax, orientation="horizontal", ticks=[1, 1.5, 2])
        cbar.ax.set_xticklabels(["1.00 · best", "1.50", "≥2 · capped"])
        cbar.ax.tick_params(length=0, pad=6)
        cbar.outline.set_visible(False)
        _family_legend(fig, order, theme)
        return _save(fig, out_path, theme)


def make_quality_vs_runtime(out_path, theme="light", *, batch=None):
    """Plot algorithm means and a step frontier from paper figure 1's feasible runs."""
    b = load_batch018() if batch is None else batch
    df = b.scaling[(b.scaling["num_exams"] == 1000) & b.scaling["feasible"]]
    means = df.groupby("algorithm").agg(runtime=("runtime", "mean"), penalty=("soft_penalty", "mean"))
    # Offsets in points, tuned for this fixed synthetic batch and the 1600 px canvas.
    offsets = {
        "Great Deluge": (0, 13, "center"), "HHO+": (10, 10, "left"),
        "ALNS": (10, 10, "left"), "Multi-Neighbourhood SA": (10, 12, "left"),
        "LAHC": (10, -12, "left"), "Genetic Algorithm": (10, 12, "left"),
        "CP-SAT B&B": (0, -17, "center"), "GVNS": (0, -17, "center"),
        "Kempe Chain": (0, 13, "center"), "Tabu Search": (12, -1, "left"),
        "WOA": (10, 6, "left"), "ABC": (10, -13, "left"),
    }

    with plt.rc_context(readme_style(theme)):
        fig, ax = _canvas(theme, "Quality vs runtime",
                          "Synthetic instance with 1,000 exams; mean of three seeds; lower and left is better",
                          height=5.5)
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14)
        points = means[["runtime", "penalty"]].to_numpy()
        front = points[_pareto_front(points)]
        ax.step(front[:, 0], front[:, 1], where="post", linewidth=0.8,
                color=THEMES[theme]["secondary"], zorder=2)
        for algorithm, row in means.iterrows():
            ax.scatter(row.runtime, row.penalty, s=45,
                       color=FAMILY_COLORS[theme][algo_family(algorithm)],
                       edgecolor=THEMES[theme]["background"], linewidth=0.8, zorder=3)
            dx, dy, align = offsets[algorithm]
            ax.annotate(algo_short(algorithm), (row.runtime, row.penalty), xytext=(dx, dy),
                        textcoords="offset points", ha=align, va="center", fontsize=9.5,
                        color=THEMES[theme]["secondary"], zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(1.5, 160)
        ax.set_ylim(0, 68000)
        ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([0, 20000, 40000, 60000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.tick_params(axis="y", colors=THEMES[theme]["muted"])
        ax.set_xlabel("Mean runtime (seconds)", labelpad=10)
        ax.set_ylabel("Mean soft penalty", labelpad=8)
        ax.grid(axis="y")
        _family_legend(fig, means.index, theme)
        return _save(fig, out_path, theme)


def make_iteration_sensitivity(out_path, theme="light", *, batch=None):
    """Plot only the iteration-budget sensitivity from paper figure 5's left panel."""
    b = load_batch018() if batch is None else batch
    df = b.sweep[b.sweep["feasible"]].copy()
    df["algorithm"] = df["algorithm"].map(lambda a: ALGO_CANONICAL.get(a, a))
    df = df[df["param_col"].map(_canonical_param) == "iters"]
    rows = [(algorithm, _sensitivity(sub)) for algorithm, sub in df.groupby("algorithm")]
    rows = sorted(((a, v) for a, v in rows if np.isfinite(v)), key=lambda row: row[1], reverse=True)
    algorithms, values = zip(*rows)

    with plt.rc_context(readme_style(theme)):
        fig, ax = _canvas(theme, "Sensitivity to the iteration budget",
                          "Larger values mean more variation in quality across the tested budgets",
                          height=5.3)
        y = np.arange(len(rows))
        ax.barh(y, values, height=0.7,
                color=[FAMILY_COLORS[theme][algo_family(a)] for a in algorithms])
        ax.set_yticks(y, [algo_short(a) for a in algorithms])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("Sensitivity: (highest − lowest) / mean soft penalty", labelpad=10)
        ax.grid(axis="x")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (value, i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=THEMES[theme]["secondary"])
        _family_legend(fig, algorithms, theme)
        return _save(fig, out_path, theme)


FIGURES = {
    "leaderboard": make_leaderboard,
    "per-set-heatmap": make_per_set_heatmap,
    "quality-vs-runtime": make_quality_vs_runtime,
    "iteration-sensitivity": make_iteration_sensitivity,
}
