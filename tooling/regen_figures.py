#!/usr/bin/env python3
"""Regenerate every paper figure from a batch's ``run_log.csv``.

Used by ``make reproduce`` and by whoever grades the project. Reads the
long-form log written by ``utils.results_logger``, splits synthetic from
public ITC 2007 rows for the scaling plots, and writes PNGs to ``--out``.

Usage::

    python -m tooling.regen_figures --from results/batch_017_tunning8 --out graphs/

Every figure name is fixed so the paper's LaTeX can reference them by path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.plots import (
    plot_algo_bars,
    plot_algo_boxes,
    plot_algo_heatmap,
    plot_algo_scatter,
    plot_feasibility_rates,
    plot_line_across_datasets,
    plot_runtime_vs_quality,
    plot_scaling,
    plot_soft_breakdown,
)


def _render(label: str, fn, /, *args, **kwargs) -> None:
    """Call a plot function with ``save_path=`` and log the outcome.

    Every plot function in ``utils.plots`` accepts ``save_path`` and
    handles its own framework-specific write (Plotly vs matplotlib), so
    we don't have to branch on the figure type here.
    """
    try:
        fn(*args, **kwargs)
        print(f"wrote {label}")
    except Exception as e:  # noqa: BLE001 — intentional per-figure tolerance
        print(f"skip  {label} ({e.__class__.__name__}: {e})")


def regenerate(run_log: Path, out_dir: Path) -> None:
    df = pd.read_csv(run_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = lambda name: str(out_dir / name)

    synth = df[df["dataset"].str.startswith("synthetic", na=False)]
    public = df[df["dataset"].str.startswith("exam_comp_set", na=False)]

    _render("algo_bars.png",         plot_algo_bars,         df, save_path=p("algo_bars.png"))
    _render("algo_boxes.png",        plot_algo_boxes,        df, save_path=p("algo_boxes.png"))
    _render("algo_heatmap.png",      plot_algo_heatmap,      df, save_path=p("algo_heatmap.png"))
    _render("algo_scatter.png",      plot_algo_scatter,      df, save_path=p("algo_scatter.png"))
    _render("pareto.png",            plot_runtime_vs_quality, df, save_path=p("pareto.png"))
    _render("feasibility_rate.png",  plot_feasibility_rates, df, save_path=p("feasibility_rate.png"))
    _render("soft_breakdown.png",    plot_soft_breakdown,    df, save_path=p("soft_breakdown.png"))

    # Family-faceted companions. Each splits the same data into 4 subplots
    # (Construction / Trajectory / Population / Exact), which reads far
    # cleaner than 13 lines overlaid. The paper can pick whichever view fits.
    _render("pareto_by_family.png", plot_runtime_vs_quality, df,
            by_family=True, save_path=p("pareto_by_family.png"))
    if not public.empty:
        _render("soft_across_by_family.png", plot_line_across_datasets, public,
                metric="soft_penalty", by_family=True,
                save_path=p("soft_across_by_family.png"))

    if not synth.empty:
        _render("runtime_synthetic.png", plot_scaling, synth,
                x_col="num_exams", y_col="runtime",
                title="Runtime scaling — synthetic instances",
                save_path=p("runtime_synthetic.png"))
    if not public.empty:
        _render("runtime_public.png", plot_scaling, public,
                x_col="num_exams", y_col="runtime",
                title="Runtime scaling — ITC 2007 public sets",
                save_path=p("runtime_public.png"))
        _render("runtime_public_by_family.png", plot_scaling, public,
                x_col="num_exams", y_col="runtime", by_family=True,
                title="Runtime scaling — by family",
                save_path=p("runtime_public_by_family.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="Batch directory containing run_log.csv")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for figures")
    args = ap.parse_args()

    log = args.src / "run_log.csv"
    if not log.is_file():
        raise SystemExit(f"run_log.csv not found in {args.src}")
    regenerate(log, args.out)


if __name__ == "__main__":
    main()
