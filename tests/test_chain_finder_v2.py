"""Unit tests for chain-finder v2 improvements.

Covers the 10 requirements from the v2 spec:
    1. Stale cleanup includes chains_sh
    2. Thread oversubscription cap
    3. Partial-credit chain results
    4. Dual chain cache keys (params + sequence)
    5. Prefix .sln cache
    6. Adaptive SH eta schedule
    7. Step-level early stop
    8. Max chain length = 10 (configurable)
    9. No adjacent duplicates in random_chain
    10. Partial-credit 2.5% penalty
    + Crossover op, proven seed expansion

Tests mock subprocess calls where possible to stay fast (<5s total).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. Stale cleanup
# ──────────────────────────────────────────────────────────────────────

def test_stale_cleanup_removes_chains_sh(monkeypatch, tmp_path):
    """Fresh (non-resume) AutoTuner must clean chains_sh/ along with
    the other subdirs. Current code only lists 'chains' (typo bug B1)."""
    import tooling.tuner.binary as _bin
    monkeypatch.setattr(_bin, 'find_or_build_binary', lambda: '/bin/true')

    # Fake a prior run layout
    out = tmp_path / 'run'
    out.mkdir()
    for sub in ('screen', 'chains_sh', 'param_tune', 'final'):
        (out / sub).mkdir()
    (out / 'chains_sh' / '_prefix_cache').mkdir()
    (out / 'checkpoint.json').write_text('{}')

    # Need at least 1 dataset file for construction
    ds = tmp_path / 'dummy.exam'
    ds.write_text('dummy')

    from tooling.tuner.core import AutoTuner
    AutoTuner(datasets=[str(ds)], output_dir=str(out), resume=False)

    assert not (out / 'chains_sh').exists(), 'chains_sh must be removed on fresh run'
    assert not (out / 'screen').exists()
    assert not (out / 'param_tune').exists()
    assert not (out / 'final').exists()
    assert not (out / 'checkpoint.json').exists()


# ──────────────────────────────────────────────────────────────────────
# 2. Oversubscription cap
# ──────────────────────────────────────────────────────────────────────

def test_cap_workers_clamps_to_max():
    from tooling.tuner.core import _cap_workers
    assert _cap_workers(9, 6) == 6
    assert _cap_workers(3, 6) == 3
    assert _cap_workers(0, 6) == 1   # floor at 1
    assert _cap_workers(-5, 6) == 1  # defensive floor


# ──────────────────────────────────────────────────────────────────────
# 3. Partial-credit chain results
# ──────────────────────────────────────────────────────────────────────

def _make_fake_popen(responses):
    """Factory returning a FakeProc class that replays `responses` in order.

    Each response is either:
        - a dict -> returncode=0, stdout is JSON array with this dict
        - 'timeout' -> raises TimeoutExpired on communicate
        - 'nonzero' -> returncode=1
        - 'bad_json' -> stdout isn't JSON
    """
    idx = [0]

    class FakeProc:
        def __init__(self, cmd, *a, **kw):
            self.pid = os.getpid()  # use real pid so /proc/$pid/status exists
            self._cmd = cmd
            r = responses[min(idx[0], len(responses) - 1)]
            idx[0] += 1
            self._r = r
            self.returncode = 1 if r == 'nonzero' else 0

        def communicate(self, timeout=None):
            import subprocess
            if self._r == 'timeout':
                raise subprocess.TimeoutExpired(self._cmd, timeout or 0)
            if self._r == 'nonzero':
                return ('', 'fake error')
            if self._r == 'bad_json':
                return ('not json', '')
            return (json.dumps([self._r]), '')

        def kill(self):
            pass

    return FakeProc


def test_partial_credit_returns_truncated_on_step1_timeout(monkeypatch, tmp_path):
    """Chain [sa, gd]: step 0 succeeds, step 1 times out → result is step 0's."""
    import algorithms.cpp_bridge as bridge

    responses = [
        {'soft_penalty': 500, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
        'timeout',
    ]
    monkeypatch.setattr(bridge.subprocess, 'Popen', _make_fake_popen(responses))
    # Make binary existence check pass
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p or p.endswith('.sln')
                        else os.path.isfile(p))

    work = tmp_path / 'work'
    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {'sa_iters': 100}), ('gd', {'gd_iters': 100})],
        seed=42,
        work_dir=str(work),
        allow_partial=True,
    )
    assert result is not None, 'should return partial result not None'
    assert result.get('chain_truncated') is True
    assert result.get('truncated_at_step') == 1
    assert result.get('failure_reason') == 'timeout'
    assert result['soft_penalty'] == 500


def test_partial_credit_off_returns_none(monkeypatch, tmp_path):
    """With allow_partial=False (default), step 1 timeout → None."""
    import algorithms.cpp_bridge as bridge

    responses = [
        {'soft_penalty': 500, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
        'timeout',
    ]
    monkeypatch.setattr(bridge.subprocess, 'Popen', _make_fake_popen(responses))
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p or p.endswith('.sln')
                        else os.path.isfile(p))

    work = tmp_path / 'work'
    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {}), ('gd', {})],
        seed=42,
        work_dir=str(work),
        allow_partial=False,
    )
    assert result is None


def test_partial_credit_step0_failure_returns_none(monkeypatch, tmp_path):
    """Even with allow_partial=True, step 0 failure gives None (no successful step to report)."""
    import algorithms.cpp_bridge as bridge

    monkeypatch.setattr(bridge.subprocess, 'Popen',
                        _make_fake_popen(['timeout']))
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p else os.path.isfile(p))

    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {}), ('gd', {})],
        seed=42,
        work_dir=str(tmp_path / 'work'),
        allow_partial=True,
    )
    assert result is None


def test_truncation_penalty_applied():
    """chain_truncated results get +2.5% multiplicative penalty in compute_score."""
    from tooling.tuner.eval import compute_score

    clean = {'soft_penalty': 1000, 'hard_violations': 0}
    truncated = {'soft_penalty': 1000, 'hard_violations': 0,
                 'chain_truncated': True}
    assert compute_score(clean) == 1000.0
    assert compute_score(truncated) == pytest.approx(1025.0)  # +2.5%


# ──────────────────────────────────────────────────────────────────────
# 4. Dual chain cache keys
# ──────────────────────────────────────────────────────────────────────

def test_chainseq_key_same_for_same_sequence_different_params():
    from tooling.eval_cache import EvalCache
    a = [('sa', {'sa_iters': 5000}), ('gd', {'gd_iters': 5000})]
    b = [('sa', {'sa_iters': 10000}), ('gd', {'gd_iters': 1000})]
    c = [('sa', {}), ('tabu', {})]

    assert EvalCache.chainseq_key('ds.exam', 42, a) == \
           EvalCache.chainseq_key('ds.exam', 42, b)
    assert EvalCache.chainseq_key('ds.exam', 42, a) != \
           EvalCache.chainseq_key('ds.exam', 42, c)
    # params-aware key still distinguishes a vs b
    assert EvalCache.chain_key('ds.exam', 42, a) != \
           EvalCache.chain_key('ds.exam', 42, b)


# ──────────────────────────────────────────────────────────────────────
# 5. Prefix .sln cache
# ──────────────────────────────────────────────────────────────────────

def test_prefix_cache_store_and_lookup(tmp_path):
    from tooling.chain_prefix_cache import PrefixCache

    pc = PrefixCache(cache_dir=str(tmp_path / 'pc'))
    sln = tmp_path / 'fake.sln'
    sln.write_text('solution content')

    pc.store(
        prefix_steps=[('sa', {'sa_iters': 100}), ('gd', {'gd_iters': 100})],
        dataset='ds.exam', seed=42,
        sln_path=str(sln),
        result={'soft_penalty': 500, 'hard_violations': 0, 'runtime': 10.0},
    )

    hit = pc.lookup(
        prefix_steps=[('sa', {'sa_iters': 100}), ('gd', {'gd_iters': 100})],
        dataset='ds.exam', seed=42,
    )
    assert hit is not None
    assert hit['soft_penalty'] == 500
    assert os.path.isfile(hit['sln_path'])


def test_prefix_cache_misses_on_different_params(tmp_path):
    """Prefix hash includes params, so [sa, i=100] ≠ [sa, i=200]."""
    from tooling.chain_prefix_cache import PrefixCache

    pc = PrefixCache(cache_dir=str(tmp_path / 'pc'))
    sln = tmp_path / 'fake.sln'
    sln.write_text('x')
    pc.store([('sa', {'sa_iters': 100})], 'ds.exam', 42, str(sln),
             {'soft_penalty': 500})
    # Different params → miss
    assert pc.lookup([('sa', {'sa_iters': 200})], 'ds.exam', 42) is None


def test_prefix_cache_lru_evicts(tmp_path):
    from tooling.chain_prefix_cache import PrefixCache

    pc = PrefixCache(cache_dir=str(tmp_path / 'pc'), max_entries=3)
    for i in range(5):
        sln = tmp_path / f'step{i}.sln'
        sln.write_text(f'step{i}')
        pc.store([(f'a{i}', {})], 'ds.exam', 42, str(sln),
                 {'soft_penalty': i})
        import time as _t
        _t.sleep(0.01)  # ensure mtime ordering

    # Oldest 2 evicted
    assert pc.lookup([('a0', {})], 'ds.exam', 42) is None
    assert pc.lookup([('a4', {})], 'ds.exam', 42) is not None


def test_prefix_cache_disabled_on_low_disk(tmp_path, monkeypatch):
    """When free disk < 5GB, PrefixCache.check_disk returns False."""
    from tooling.chain_prefix_cache import PrefixCache
    # Simulate 1GB free
    fake_usage = type('U', (), {'free': 1 * 1024 ** 3})()
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: fake_usage)
    assert PrefixCache.check_disk(str(tmp_path)) is False


# ──────────────────────────────────────────────────────────────────────
# 6. Adaptive SH eta schedule
# ──────────────────────────────────────────────────────────────────────

def test_eta_schedule_for_pop():
    from tooling.tuner.core import _eta_schedule_for_pop
    assert _eta_schedule_for_pop(4) == [2, 2, 2]
    assert _eta_schedule_for_pop(8) == [2, 2, 2]
    assert _eta_schedule_for_pop(12) == [3, 2, 2]
    assert _eta_schedule_for_pop(16) == [3, 2, 2]
    assert _eta_schedule_for_pop(32) == [4, 3, 2]


def test_successive_halving_eta_schedule():
    """eta_schedule overrides flat eta per rung."""
    from tooling.successive_halving import successive_halving

    # 8 candidates, rung 0 scores them 0..7, rung 1 scores same ranking
    call_count = [0]
    def eval_fn(cand, rung_idx, fidelity):
        call_count[0] += 1
        return cand  # scores are just the candidate ints

    candidates = list(range(8))
    rungs = [('cheap',), ('full',)]
    # eta_schedule [4, 1] means: after rung 0, keep 8//4=2 candidates
    winner, score, hist = successive_halving(
        candidates, eval_fn, rungs, eta_schedule=[4, 1])
    # 8 at rung 0 + 2 at rung 1 = 10 evaluations
    assert call_count[0] == 10


# ──────────────────────────────────────────────────────────────────────
# 7. Step-level early stop
# ──────────────────────────────────────────────────────────────────────

def test_step_early_stop_aborts_on_high_soft(monkeypatch, tmp_path):
    """Step 1 returns soft=2000 > threshold=500 → abort, step 2 never runs."""
    import algorithms.cpp_bridge as bridge

    responses = [
        {'soft_penalty': 300, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
        {'soft_penalty': 2000, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
        {'soft_penalty': 100, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},  # should never be called
    ]
    popen_class = _make_fake_popen(responses)
    monkeypatch.setattr(bridge.subprocess, 'Popen', popen_class)
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p or p.endswith('.sln')
                        else os.path.isfile(p))

    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {}), ('gd', {}), ('tabu', {})],
        seed=42,
        work_dir=str(tmp_path / 'work'),
        allow_partial=True,
        abort_threshold_soft=500,
    )
    assert result is not None
    assert result.get('chain_truncated') is True
    assert result.get('failure_reason') == 'abort_unworthy'
    assert result.get('truncated_at_step') == 2  # aborted after step 1 (0-idx) completed


def test_step_early_stop_skips_last_step(monkeypatch, tmp_path):
    """Don't abort if we're already at the last step — pointless."""
    import algorithms.cpp_bridge as bridge

    responses = [
        {'soft_penalty': 300, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
        {'soft_penalty': 2000, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
    ]
    monkeypatch.setattr(bridge.subprocess, 'Popen', _make_fake_popen(responses))
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p or p.endswith('.sln')
                        else os.path.isfile(p))

    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {}), ('gd', {})],  # only 2 steps
        seed=42,
        work_dir=str(tmp_path / 'work'),
        allow_partial=True,
        abort_threshold_soft=500,
    )
    # Last step is step 1 — even though soft=2000 > threshold, don't abort
    assert result is not None
    assert not result.get('chain_truncated', False)
    assert result['soft_penalty'] == 2000


def test_step_early_stop_skips_infeasible(monkeypatch, tmp_path):
    """Infeasible intermediate steps should not be aborted — later repair
    steps might fix them."""
    import algorithms.cpp_bridge as bridge

    responses = [
        {'soft_penalty': 100, 'hard_violations': 5, 'runtime': 1.0,  # infeasible
         'evaluation': '{"soft_breakdown":{}}'},
        {'soft_penalty': 200, 'hard_violations': 0, 'runtime': 1.0,
         'evaluation': '{"soft_breakdown":{}}'},
    ]
    monkeypatch.setattr(bridge.subprocess, 'Popen', _make_fake_popen(responses))
    monkeypatch.setattr(bridge.os.path, 'isfile',
                        lambda p: True if 'exam_solver' in p or p.endswith('.sln')
                        else os.path.isfile(p))

    result = bridge.run_chain(
        'dummy.exam',
        [('sa', {}), ('gd', {})],
        seed=42,
        work_dir=str(tmp_path / 'work'),
        allow_partial=True,
        abort_threshold_soft=50,  # step 0 soft=100 > threshold but infeasible
    )
    assert result is not None
    assert not result.get('chain_truncated', False)


# ──────────────────────────────────────────────────────────────────────
# 8 & 9. Max chain length + adjacent-dup guard
# ──────────────────────────────────────────────────────────────────────

def test_max_chain_len_is_10():
    from tooling.tuner.search_spaces import MAX_CHAIN_LEN
    assert MAX_CHAIN_LEN == 10


def test_random_chain_respects_length_cap():
    from tooling.tuner.sampling import random_chain
    rng = random.Random(0)
    chain = random_chain(['sa', 'gd', 'tabu', 'kempe'], {}, rng, length=15)
    assert len(chain) <= 10


def test_random_chain_no_adjacent_duplicates():
    from tooling.tuner.sampling import random_chain
    rng = random.Random(0)
    for _ in range(50):
        c = random_chain(['sa', 'gd', 'tabu'], {}, rng,
                         length=8, allow_duplicates=False)
        algos = [a for a, _ in c]
        for i in range(1, len(algos)):
            assert algos[i] != algos[i-1], f'adjacent dup at {i}: {algos}'


def test_random_chain_allow_duplicates_may_produce_dups():
    """With allow_duplicates=True, dups are possible (but not required)."""
    from tooling.tuner.sampling import random_chain
    rng = random.Random(0)
    # With only 2 algos and length 10, dups are near-certain statistically
    c = random_chain(['sa', 'gd'], {}, rng, length=10, allow_duplicates=True)
    assert len(c) == 10  # just check it ran — duplicates are statistically
                          # certain but we don't test the exact arrangement


# ──────────────────────────────────────────────────────────────────────
# Crossover
# ──────────────────────────────────────────────────────────────────────

def test_crossover_produces_valid_child():
    from tooling.tuner.sampling import crossover
    rng = random.Random(0)
    a = [('sa', {}), ('gd', {}), ('tabu', {}), ('kempe', {})]
    b = [('alns', {}), ('vns', {}), ('hho', {}), ('lahc', {})]
    for _ in range(20):
        child = crossover(a, b, rng, allow_duplicates=False)
        assert 2 <= len(child) <= 10
        algos = [step[0] for step in child]
        for i in range(1, len(algos)):
            assert algos[i] != algos[i-1], f'adjacent dup in crossover: {algos}'


def test_crossover_length_clamp():
    """Crossover of two long chains must not exceed MAX_CHAIN_LEN."""
    from tooling.tuner.sampling import crossover, MAX_CHAIN_LEN
    rng = random.Random(0)
    # Build two 8-step chains with no internal dups
    pool_a = ['sa', 'gd', 'tabu', 'kempe', 'lahc', 'vns', 'hho', 'alns']
    pool_b = ['alns', 'hho', 'vns', 'lahc', 'kempe', 'tabu', 'gd', 'sa']
    a = [(x, {}) for x in pool_a]
    b = [(x, {}) for x in pool_b]
    child = crossover(a, b, rng, allow_duplicates=True)
    assert len(child) <= MAX_CHAIN_LEN


def test_vary_chain_uses_crossover_when_rate_high():
    """With crossover_rate=1.0, vary_chain always crosses over."""
    from tooling.tuner.sampling import vary_chain
    rng = random.Random(42)
    parent = [('sa', {}), ('gd', {})]
    other = [('tabu', {}), ('kempe', {})]
    result = vary_chain(parent, ['sa', 'gd', 'tabu', 'kempe'], {},
                        rng, survivors=[parent, other],
                        crossover_rate=1.0)
    assert 2 <= len(result) <= 10


# ──────────────────────────────────────────────────────────────────────
# Proven seed expansion
# ──────────────────────────────────────────────────────────────────────

def test_proven_chains_include_champion():
    """User's winning chain kempe→alns→kempe→tabu must be seeded."""
    from tooling.tuner.search_spaces import PROVEN_CHAIN_TEMPLATES
    assert ('kempe', 'alns', 'kempe', 'tabu') in PROVEN_CHAIN_TEMPLATES


def test_proven_chains_include_recent_algos():
    """ALNS, VNS, HHO must appear in at least one proven template."""
    from tooling.tuner.search_spaces import PROVEN_CHAIN_TEMPLATES
    all_algos = set()
    for template in PROVEN_CHAIN_TEMPLATES:
        all_algos.update(template)
    assert 'alns' in all_algos, 'ALNS missing from proven seeds'
    assert 'vns' in all_algos, 'VNS missing from proven seeds'
    assert 'hho' in all_algos, 'HHO missing from proven seeds'
