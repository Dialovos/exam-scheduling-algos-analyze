"""Tests for tooling.successive_halving."""
import pytest

from tooling.successive_halving import successive_halving


def test_sh_promotes_best_candidates():
    # 8 candidates; fitness = candidate_id (lower is better).
    # Rungs: (2 pulls, 1) -> halve -> (2 pulls, 2) -> halve -> (2 pulls, 3)
    # Expected winner: id=0
    candidates = list(range(8))
    calls = {'n': 0}

    def eval_fn(candidate, rung_idx, fidelity):
        calls['n'] += 1
        # fidelity unused here, score is deterministic per candidate
        return float(candidate)

    rungs = [(1, 1), (2, 2), (3, 3)]  # (n_seeds, n_datasets) at each rung
    winner, score, history = successive_halving(candidates, eval_fn, rungs)
    assert winner == 0
    assert score == 0.0


def test_sh_reduces_total_evaluations():
    # If we had no halving, cost = 8 * 3 rungs = 24 entries.
    # With halving by 2 per rung: 8 + 4 + 2 = 14 entries.
    candidates = list(range(8))
    fake_scores = {i: float(i) for i in range(8)}
    call_log = []

    def eval_fn(candidate, rung_idx, fidelity):
        call_log.append((candidate, rung_idx, fidelity))
        return fake_scores[candidate]

    rungs = [(1, 1), (2, 2), (3, 3)]
    winner, score, history = successive_halving(candidates, eval_fn, rungs)
    assert len(call_log) == 8 + 4 + 2  # 14 calls total


def test_sh_single_rung_eval_all():
    candidates = ['a', 'b', 'c']
    scores = {'a': 3.0, 'b': 1.0, 'c': 2.0}
    def eval_fn(c, r, f): return scores[c]
    winner, score, hist = successive_halving(candidates, eval_fn, [(1, 1)])
    assert winner == 'b'
    assert score == 1.0


def test_sh_history_records_all_evaluations():
    candidates = list(range(4))
    def eval_fn(c, r, f): return float(c)
    winner, score, hist = successive_halving(candidates, eval_fn, [(1, 1), (2, 2)])
    # rung 0: 4 evals; rung 1: 2 evals (top half)
    assert len(hist) == 4 + 2
    # Each entry is (candidate, rung_idx, score)
    assert all(len(e) == 3 for e in hist)


def test_sh_handles_infeasible_scores():
    # If all candidates return inf, still return something (the "first" best)
    candidates = list(range(4))
    def eval_fn(c, r, f): return float('inf')
    winner, score, hist = successive_halving(candidates, eval_fn, [(1, 1), (2, 2)])
    assert winner in candidates
    assert score == float('inf')


def test_sh_empty_candidates_raises():
    with pytest.raises(ValueError):
        successive_halving([], lambda c, r, f: 0.0, [(1, 1)])
