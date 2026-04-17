"""Smoke tests for the ``tooling.tuner`` package split.

The tuner used to be a single 1421-line file. These tests guarantee the
split preserved every name the legacy callers depend on:

    * main.py               → AutoTuner
    * tooling.optimizers    → SEARCH_SPACES
    * exam_scheduling.ipynb → auto_tuner module object

If any of those break, the tuner is unusable and the paper pipeline
silently regresses — cheap insurance for a one-time-per-decade refactor.
"""
from __future__ import annotations

import importlib


def test_tuner_package_exposes_core_api():
    """The headline names must be importable from the top-level package."""
    import tooling.tuner as t
    for name in (
        "AutoTuner", "main", "SEARCH_SPACES", "DEFAULT_PARAMS",
        "TUNABLE_ALGOS", "EVAL_SEEDS", "compute_score",
        "run_single_algo", "run_chain", "Checkpoint",
        "generate_synthetic_dataset",
    ):
        assert hasattr(t, name), f"tooling.tuner is missing {name!r}"


def test_search_spaces_covers_expected_algos():
    from tooling.tuner import SEARCH_SPACES, TUNABLE_ALGOS
    expected = {"tabu", "sa", "kempe", "alns", "gd", "abc", "ga", "lahc", "woa", "vns"}
    assert set(TUNABLE_ALGOS) == expected
    # Every entry should be a dict mapping knob -> (lo, hi, scale)
    for algo, space in SEARCH_SPACES.items():
        assert space, f"{algo} has empty search space"
        for knob, spec in space.items():
            assert len(spec) == 3 and spec[2] in ("log", "int"), \
                f"{algo}.{knob} malformed: {spec}"


def test_auto_tuner_shim_re_exports_everything():
    """Existing `from tooling.auto_tuner import X` callers keep working."""
    at = importlib.reload(importlib.import_module("tooling.auto_tuner"))
    for name in (
        "AutoTuner", "main", "SEARCH_SPACES", "DEFAULT_PARAMS",
        "compute_score", "run_single_algo",
    ):
        assert hasattr(at, name), f"shim dropped {name!r}"
    # Class identity: shim's AutoTuner IS the real one
    from tooling.tuner.core import AutoTuner as Real
    assert at.AutoTuner is Real


def test_compute_score_feasibility_first():
    from tooling.tuner.eval import compute_score
    assert compute_score(None) == float('inf')
    # No hard violations → soft penalty is the score
    assert compute_score({"hard_violations": 0, "soft_penalty": 1234}) == 1234
    # Any hard violation → score jumps above 1e9
    bad = compute_score({"hard_violations": 1, "soft_penalty": 100})
    assert bad >= 1e9
    # More violations → higher score
    worse = compute_score({"hard_violations": 5, "soft_penalty": 100})
    assert worse > bad


def test_sampling_respects_bounds():
    import random
    from tooling.tuner.sampling import sample_random, perturb
    rng = random.Random(42)
    for algo in ("tabu", "sa", "ga"):
        params = sample_random(algo, rng)
        assert params, f"sample_random({algo}) returned empty"
        # Perturb stays within bounds
        for _ in range(20):
            perturbed = perturb(algo, params, rng)
            from tooling.tuner.search_spaces import SEARCH_SPACES
            for k, v in perturbed.items():
                lo, hi, _ = SEARCH_SPACES[algo][k]
                assert lo <= v <= hi, f"{algo}.{k}={v} out of [{lo}, {hi}]"
