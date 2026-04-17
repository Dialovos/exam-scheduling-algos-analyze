"""End-to-end tests for the universal 4-layer seeder.

The seeder is our "bouncer at the door" — nothing downstream gets past infeasible.
These tests exercise it through the actual C++ subprocess bridge the algos use,
because we want to catch wire-up regressions (JSON shape, CLI dispatch, etc.)
not just the internal logic.
"""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "cpp" / "build" / "exam_solver"
INSTANCE_SET1 = ROOT / "instances" / "exam_comp_set1.exam"


def _extract_json_array(stdout: str) -> list:
    """Pull the trailing JSON array out of the solver's stdout."""
    start = stdout.rfind("[")
    end = stdout.rfind("]")
    assert start >= 0 and end > start, f"no JSON array in stdout:\n{stdout}"
    return json.loads(stdout[start : end + 1])


def _run(*args, timeout=60):
    assert BINARY.exists(), f"binary not built: {BINARY} (run `make`)"
    assert INSTANCE_SET1.exists(), f"instance missing: {INSTANCE_SET1}"
    return subprocess.run(
        [str(BINARY), str(INSTANCE_SET1), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_seeder_produces_feasible_start_on_set1():
    """Layer 1 (DSatur) alone should solve set1 — it's a benign instance."""
    r = _run("--algo", "seeder", "--seed", "42")
    assert r.returncode == 0, f"bridge failed: {r.stderr}"
    results = _extract_json_array(r.stdout)
    assert len(results) == 1
    seeder = results[0]
    assert seeder["algorithm"].lower().startswith("seeder"), seeder["algorithm"]
    assert seeder["hard_violations"] == 0, (
        f"seeder must produce feasible start, got hard={seeder['hard_violations']}"
    )
    assert seeder["feasible"] is True


def test_seeder_reports_layer_used_in_json():
    """The JSON payload must carry which layer (1-4) produced the feasible start,
    so the notebook can report escalation rates per dataset."""
    r = _run("--algo", "seeder", "--seed", "42")
    assert r.returncode == 0, f"bridge failed: {r.stderr}"
    assert "seeder_layer" in r.stdout, (
        "missing seeder_layer field in JSON output — needed for reporting"
    )
    results = _extract_json_array(r.stdout)
    assert "seeder_layer" in results[0]
    layer = results[0]["seeder_layer"]
    assert 1 <= layer <= 4, f"layer out of range: {layer}"


def test_seeder_is_deterministic_per_seed():
    """Same seed → same soft penalty. If this drifts we have uninitialised RNG state
    somewhere, which is the kind of bug that silently breaks reproducibility."""
    r1 = _run("--algo", "seeder", "--seed", "7")
    r2 = _run("--algo", "seeder", "--seed", "7")
    assert r1.returncode == 0 and r2.returncode == 0
    s1 = _extract_json_array(r1.stdout)[0]
    s2 = _extract_json_array(r2.stdout)[0]
    assert s1["hard_violations"] == s2["hard_violations"]
    assert s1["soft_penalty"] == s2["soft_penalty"]
