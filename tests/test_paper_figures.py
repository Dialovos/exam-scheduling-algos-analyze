"""Smoke tests for the six paper figures."""
from __future__ import annotations

import pytest


def _assert_png_ok(path, min_bytes=10_000):
    assert path.exists(), f"Figure not written: {path}"
    assert path.stat().st_size > min_bytes, (
        f"Figure {path} is only {path.stat().st_size} bytes; expected > {min_bytes}"
    )


def test_fig2_family_heatmap_smoke(tmp_path):
    from utils.plots.paper.fig2_family_heatmap import make_fig2
    out = tmp_path / "fig2.png"
    make_fig2(out)
    _assert_png_ok(out)


def test_fig1_pareto_smoke(tmp_path):
    from utils.plots.paper.fig1_pareto import make_fig1
    out = tmp_path / "fig1.png"
    make_fig1(out)
    _assert_png_ok(out, min_bytes=30_000)


def test_fig6_ip_vs_heuristic_smoke(tmp_path):
    from utils.plots.paper.fig6_ip_vs_heuristic import make_fig6
    out = tmp_path / "fig6.png"
    make_fig6(out)
    _assert_png_ok(out, min_bytes=25_000)


def test_fig4_scaling_smoke(tmp_path):
    from utils.plots.paper.fig4_scaling import make_fig4
    out = tmp_path / "fig4.png"
    make_fig4(out)
    _assert_png_ok(out, min_bytes=25_000)


def test_fig5_sensitivity_smoke(tmp_path):
    from utils.plots.paper.fig5_sensitivity import make_fig5
    out = tmp_path / "fig5.png"
    make_fig5(out)
    _assert_png_ok(out, min_bytes=20_000)


def test_fig3_chain_methodology_smoke(tmp_path):
    from utils.plots.paper.fig3_chain_methodology import make_fig3
    out = tmp_path / "fig3.png"
    make_fig3(out)
    _assert_png_ok(out, min_bytes=25_000)


def test_fig7_gap_heatmap_smoke(tmp_path):
    from utils.plots.paper.fig7_gap_heatmap import make_fig7
    out = tmp_path / "fig7.png"
    make_fig7(out)
    _assert_png_ok(out, min_bytes=25_000)


def test_fig8_gap_leaderboard_smoke(tmp_path):
    from utils.plots.paper.fig8_gap_leaderboard import make_fig8
    out = tmp_path / "fig8.png"
    make_fig8(out)
    _assert_png_ok(out, min_bytes=20_000)
