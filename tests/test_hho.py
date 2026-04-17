"""Smoke tests for the re-introduced HHO+ algorithm.

These are cheap sanity checks — we're not benchmarking HHO+ here, just
confirming the subprocess bridge, CLI args, and the core loop don't crash.
Full quality validation belongs in the notebook batch runs.
"""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "cpp" / "build" / "exam_solver"
INSTANCE_SET1 = ROOT / "instances" / "exam_comp_set1.exam"


def _extract_json_array(stdout: str) -> list:
    start = stdout.rfind("[")
    end = stdout.rfind("]")
    assert start >= 0 and end > start, f"no JSON array in stdout:\n{stdout}"
    return json.loads(stdout[start : end + 1])


def _run_hho(seed: int, iters: int = 30, pop: int = 8, timeout: int = 60):
    assert BINARY.exists(), f"binary not built: {BINARY}"
    return subprocess.run(
        [
            str(BINARY), str(INSTANCE_SET1),
            "--algo", "hho",
            "--hho-iters", str(iters),
            "--hho-pop", str(pop),
            "--seed", str(seed),
        ],
        capture_output=True, text=True, timeout=timeout,
    )


def test_hho_produces_feasible_solution():
    """HHO+ warm-starts from Seeder, so even a handful of iters must stay feasible."""
    r = _run_hho(seed=42, iters=30, pop=6)
    assert r.returncode == 0, f"bridge failed: {r.stderr}"
    out = _extract_json_array(r.stdout)
    assert len(out) == 1
    res = out[0]
    assert res["algorithm"] == "HHO+"
    assert res["feasible"] is True
    assert res["hard_violations"] == 0


def test_hho_improves_or_matches_seed_on_set1():
    """Very low bar: HHO+ should never land worse than its seeder init.
    Catches bugs where the accept logic is flipped or rollbacks don't restore."""
    # Baseline: seeder alone.
    seed_proc = subprocess.run(
        [str(BINARY), str(INSTANCE_SET1), "--algo", "seeder", "--seed", "42"],
        capture_output=True, text=True, timeout=30,
    )
    assert seed_proc.returncode == 0
    seed_result = _extract_json_array(seed_proc.stdout)[0]

    # HHO+ with short budget.
    hho_proc = _run_hho(seed=42, iters=30, pop=6)
    assert hho_proc.returncode == 0
    hho_result = _extract_json_array(hho_proc.stdout)[0]

    # On set1 the seed is already feasible; HHO+ should keep hard=0.
    assert hho_result["hard_violations"] == 0
    # Soft should be ≤ seed's soft (we return best ever seen).
    assert hho_result["soft_penalty"] <= seed_result["soft_penalty"], (
        f"HHO+ regressed: hho_soft={hho_result['soft_penalty']} "
        f"> seed_soft={seed_result['soft_penalty']}"
    )
