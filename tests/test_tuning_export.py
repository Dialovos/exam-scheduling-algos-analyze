"""Tests for the RQ3-facing sensitivity export pipeline.

Covers:
    * ``tooling.tuning_export`` CLI-friendly JSON → CSV conversion
    * ``AutoTuner._rank_params_by_impact`` variance-based ranking fallback
    * ``utils.plots.tuning.plot_tuning_sensitivity`` heatmap + 1-D degrade
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def test_tuning_export_emits_two_tidy_csvs(tmp_path):
    from tooling.tuning_export import export

    log = [
        {
            "version": 1,
            "timestamp": "2026-04-16T10:00:00",
            "source": "auto_tuner",
            "params": {
                "kempe": {"kempe_iters": 10000},
                "tabu": {"tabu_iters": 2000, "tabu_tenure": 20},
            },
            "per_dataset_scores": {
                "exam_comp_set1": 11150.0,
                "exam_comp_set4": 26000.0,
            },
            "trial_count": 42,
            "aggregate_score": 3.297,
        }
    ]
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(log))

    algos_csv, ds_csv = export(log_path, tmp_path / "tx.csv")

    algo_rows = list(csv.DictReader(open(algos_csv)))
    ds_rows = list(csv.DictReader(open(ds_csv)))

    # 1 kempe knob + 2 tabu knobs = 3 rows
    assert len(algo_rows) == 3
    assert {r["algorithm"] for r in algo_rows} == {"kempe", "tabu"}
    assert {r["param"] for r in algo_rows} == {"kempe_iters", "tabu_iters", "tabu_tenure"}

    assert len(ds_rows) == 2
    assert {r["dataset"] for r in ds_rows} == {"exam_comp_set1", "exam_comp_set4"}
    assert all(float(r["aggregate_score"]) == 3.297 for r in ds_rows)


def test_rank_params_by_impact_prefers_higher_variance():
    """When param_history has enough signal, rank by |correlation|."""
    from tooling.tuner.core import AutoTuner

    # Bypass __init__ (which would try to compile the binary) — just test
    # the pure-function ranker on a stub instance.
    tuner = AutoTuner.__new__(AutoTuner)
    # tabu has 3 knobs: tabu_iters, tabu_tenure, tabu_patience
    tuner.param_history = []
    # Fabricate a history where tabu_iters strongly predicts score (linear)
    # and tabu_tenure is decorrelated noise.
    for i, iters in enumerate((1000, 5000, 10000, 15000, 20000)):
        tuner.param_history.append(
            ("tabu",
             {"tabu_iters": iters, "tabu_tenure": 10 + (i * 37) % 40, "tabu_patience": 500},
             100000 - iters * 3)  # lower iters → higher (worse) score
        )
    ranked = tuner._rank_params_by_impact("tabu",
                                          {"tabu_iters": 2000, "tabu_tenure": 20, "tabu_patience": 500})
    assert ranked[0] == "tabu_iters", f"expected tabu_iters first, got {ranked!r}"


def test_rank_params_falls_back_to_search_space_order():
    """Sparse history → preserve SEARCH_SPACES key order."""
    from tooling.tuner.core import AutoTuner
    tuner = AutoTuner.__new__(AutoTuner)
    tuner.param_history = []
    ranked = tuner._rank_params_by_impact("tabu",
                                          {"tabu_iters": 2000, "tabu_tenure": 20, "tabu_patience": 500})
    assert ranked == ["tabu_iters", "tabu_tenure", "tabu_patience"]


def test_plot_tuning_sensitivity_1d_fallback_does_not_crash():
    """A 1-knob sweep (param_b empty) should degrade to the curve plot."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    from utils.plots.tuning import plot_tuning_sensitivity

    rows = []
    for v in (5000, 10000, 15000):
        for s in range(3):
            rows.append({
                "algorithm": "kempe", "param_a": "kempe_iters", "value_a": v,
                "param_b": "", "value_b": "",
                "seed": s, "soft_penalty": 12000 - v / 5 + s * 100,
                "runtime": 3.0 + s * 0.1, "feasible": True,
            })
    df = pd.DataFrame(rows)
    fig = plot_tuning_sensitivity(df, algorithm="kempe")
    assert fig is not None


def test_plot_tuning_sensitivity_2d_heatmap():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    from utils.plots.tuning import plot_tuning_sensitivity

    rows = []
    for va in (20, 50, 100):
        for vb in (100, 500, 1000):
            for s in range(2):
                rows.append({
                    "algorithm": "ga", "param_a": "ga_pop", "value_a": va,
                    "param_b": "ga_iters", "value_b": vb,
                    "seed": s, "soft_penalty": 10000 - va * 20 - vb / 10 + s * 50,
                    "runtime": 2.0, "feasible": True,
                })
    df = pd.DataFrame(rows)
    fig = plot_tuning_sensitivity(df, algorithm="ga")
    assert fig is not None
    # Heatmap should have exactly one colorbar axes
    assert len(fig.axes) == 2  # main + colorbar
