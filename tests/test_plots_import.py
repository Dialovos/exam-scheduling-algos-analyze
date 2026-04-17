"""Smoke tests for the utils.plots package split.

These verify three invariants the refactor cannot break:
  1. The new utils.plots package is importable and exports the palette.
  2. Every plot function callers already depend on is reachable from
     utils.plots at top level (the notebook does `from utils.plots import *`).
  3. The old utils.plotting wildcard import still works — we keep that shim
     until the notebook is migrated so one file does not block another.
"""
from __future__ import annotations


def test_plots_package_importable():
    """utils.plots must import cleanly and expose the canonical palette."""
    import utils.plots
    assert hasattr(utils.plots, "ALGO_SHORT")
    assert hasattr(utils.plots, "ALGO_COLORS")
    assert hasattr(utils.plots, "algo_color")
    assert hasattr(utils.plots, "apply_paper_style")


def test_plots_exports_headline_functions():
    """Functions used by main.py and the notebook must stay reachable."""
    import utils.plots as p
    for name in (
        "plot_algo_bars",
        "plot_experiment_summary",
        "plot_continuous_scan",
        "plot_soft_constraint_breakdown",
        "generate_all_plots",
    ):
        assert hasattr(p, name), f"utils.plots missing {name}"


def test_backward_compat_shim():
    """Existing utils.plotting callers keep working during the migration."""
    import importlib
    import utils.plotting as pl
    importlib.reload(pl)
    assert hasattr(pl, "plot_algo_bars")
    assert hasattr(pl, "ALGO_COLORS")


def test_palette_has_new_algo_entries():
    """Seeder + HHO+ got added in this refactor — guard against regressions."""
    from utils.plots.shared import ALGO_COLORS, ALGO_SHORT
    assert "Seeder" in ALGO_COLORS
    assert "HHO+" in ALGO_SHORT
