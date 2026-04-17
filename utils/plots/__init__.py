"""utils.plots — themed plotting modules split from the old 1500-line plotting.py.

The shared palette, helper functions, and paper-style rc params all live in
:mod:`utils.plots.shared`; specific plot families live in their own submodules
so no file ever exceeds a reviewer's patience:

    * :mod:`utils.plots.comparative` — bars, boxes, radar, heatmaps, rank tables
    * :mod:`utils.plots.convergence` — scatter, scaling, convergence overlay
    * :mod:`utils.plots.breakdown`   — soft-constraint breakdown + dashboard
    * :mod:`utils.plots.tuning`      — parameter-sensitivity dual-axis plot

Everything exported here matches the historical ``utils.plotting`` API surface
so existing callers (notebook wildcard imports, ``main.py``) keep working.
"""
from __future__ import annotations

import os

# Shared primitives
from utils.plots.shared import (
    ALGO_COLORS, ALGO_MARKERS, ALGO_SHORT, ALGO_ORDER,
    SOFT_KEYS, SOFT_LABELS, SOFT_COLORS, METRIC_LABELS, HAS_MPL,
    algo_color, algo_marker, algo_short, algo_order, apply_paper_style,
    _c, _m, _short, _order, _style, _kfmt, _apply_xlabels, _save,
)

# Comparative plots
from utils.plots.comparative import (
    plot_algorithm_comparison, plot_multi_dataset_heatmap,
    plot_feasibility_rates, plot_box_comparison, plot_radar, plot_rank_table,
    plot_experiment_summary, plot_algo_bars, plot_algo_boxes,
    plot_algo_radar, plot_algo_heatmap,
)

# Convergence / scaling plots
from utils.plots.convergence import (
    plot_runtime_vs_quality, plot_scaling, plot_convergence,
    plot_line_across_datasets, plot_continuous_scan, plot_algo_scatter,
)

# Breakdown / dashboard
from utils.plots.breakdown import (
    plot_soft_breakdown, plot_soft_lines, plot_summary_dashboard,
    plot_soft_constraint_breakdown,
)

# Tuning / sensitivity
from utils.plots.tuning import plot_parameter_sensitivity, plot_tuning_sensitivity


# ── Batch-generation orchestrators ───────────────────────────────────────
# Live here rather than in a submodule because they glue every theme
# together and have no other home where they would feel less arbitrary.

def save_all_plotly(df, output_dir: str = "graphs") -> None:
    """Save all Plotly summary charts as PNG to *output_dir*.

    Suppresses ``fig.show()`` during save so the notebook doesn't show every
    chart twice (once from the build, once from the display loop).
    """
    import plotly.graph_objects as go
    os.makedirs(output_dir, exist_ok=True)
    _orig = go.Figure.show
    go.Figure.show = lambda self, *a, **k: None
    try:
        p = lambda name: os.path.join(output_dir, name)
        plot_experiment_summary(df, save_path=p("summary_lines.png"))
        plot_algo_bars(df,              save_path=p("algo_bars.png"))
        plot_algo_boxes(df,             save_path=p("algo_boxes.png"))
        plot_algo_radar(df,             save_path=p("algo_radar.png"))
        plot_algo_scatter(df,           save_path=p("algo_scatter.png"))
        plot_algo_heatmap(df,           save_path=p("algo_heatmap.png"))
    finally:
        go.Figure.show = _orig
    print(f"[Plot] Saved 6 charts to {output_dir}/")


def generate_all_plots(logger_or_df, output_dir: str = "results") -> None:
    """Render the full matplotlib suite and save everything to *output_dir*."""
    if not HAS_MPL:
        print("[Plot] matplotlib not available")
        return
    df = (logger_or_df.to_dataframe()
          if hasattr(logger_or_df, "to_dataframe") else logger_or_df)
    if df.empty:
        print("[Plot] No data"); return
    os.makedirs(output_dir, exist_ok=True)

    for ds in df["dataset"].unique():
        s = ds.replace(" ", "_")
        plot_algorithm_comparison(df, dataset=ds, metric="soft_penalty",
                                  save_path=f"{output_dir}/{s}_soft_comparison.png")
        plot_algorithm_comparison(df, dataset=ds, metric="runtime",
                                  save_path=f"{output_dir}/{s}_runtime_comparison.png")
        plot_soft_breakdown(df, dataset=ds,
                            save_path=f"{output_dir}/{s}_soft_breakdown.png")
        plot_soft_lines(df, dataset=ds,
                        save_path=f"{output_dir}/{s}_soft_lines.png")
        plot_runtime_vs_quality(df, dataset=ds,
                                save_path=f"{output_dir}/{s}_quality_vs_time.png")
        plot_box_comparison(df, dataset=ds,
                            save_path=f"{output_dir}/{s}_box_comparison.png")
        plot_radar(df, dataset=ds,
                   save_path=f"{output_dir}/{s}_radar.png")
        plot_rank_table(df, dataset=ds,
                        save_path=f"{output_dir}/{s}_ranking.png")

    if len(df["dataset"].unique()) > 1:
        plot_multi_dataset_heatmap(df, save_path=f"{output_dir}/heatmap_soft.png")
        plot_line_across_datasets(df, save_path=f"{output_dir}/line_across_datasets.png")
        plot_feasibility_rates(df, save_path=f"{output_dir}/feasibility_rates.png")

    plot_summary_dashboard(df, save_path=f"{output_dir}/dashboard.png")
    print(f"[Plot] All plots saved to {output_dir}/")


__all__ = [
    # Palette / metadata
    "ALGO_COLORS", "ALGO_MARKERS", "ALGO_SHORT", "ALGO_ORDER",
    "SOFT_KEYS", "SOFT_LABELS", "SOFT_COLORS", "METRIC_LABELS",
    "algo_color", "algo_marker", "algo_short", "algo_order", "apply_paper_style",
    # Comparative
    "plot_algorithm_comparison", "plot_multi_dataset_heatmap",
    "plot_feasibility_rates", "plot_box_comparison", "plot_radar", "plot_rank_table",
    "plot_experiment_summary", "plot_algo_bars", "plot_algo_boxes",
    "plot_algo_radar", "plot_algo_heatmap",
    # Convergence
    "plot_runtime_vs_quality", "plot_scaling", "plot_convergence",
    "plot_line_across_datasets", "plot_continuous_scan", "plot_algo_scatter",
    # Breakdown
    "plot_soft_breakdown", "plot_soft_lines", "plot_summary_dashboard",
    "plot_soft_constraint_breakdown",
    # Tuning
    "plot_parameter_sensitivity", "plot_tuning_sensitivity",
    # Orchestrators
    "generate_all_plots", "save_all_plotly",
]
