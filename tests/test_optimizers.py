"""Tests for tooling.optimizers."""
import math
import pytest

from tooling.optimizers import golden_section_search, nelder_mead, optimize_params


def test_golden_section_finds_minimum_of_parabola():
    # f(x) = (x - 3.7)^2 on [0, 10] — minimum at 3.7
    f = lambda x: (x - 3.7) ** 2
    best_x, best_y, history = golden_section_search(f, lo=0, hi=10, scale='int', n_evals=20)
    assert abs(best_x - 4) <= 1  # rounded to int, within 1 of true 3.7
    assert len(history) == 20


def test_golden_section_log_scale_on_log_parabola():
    # f(k) = (log(k) - log(5000))^2 on log scale [100, 50000]
    target = 5000
    f = lambda x: (math.log(max(x, 1)) - math.log(target)) ** 2
    best_x, best_y, history = golden_section_search(f, lo=100, hi=50000, scale='log', n_evals=15)
    # Log-scale convergence: should be within 50% of target
    assert 2500 <= best_x <= 10000


def test_golden_section_respects_eval_budget():
    calls = [0]

    def f(x):
        calls[0] += 1
        return x * x

    golden_section_search(f, lo=-10, hi=10, scale='int', n_evals=7)
    # Budget is an upper bound — cache dedup can reduce real solver calls.
    assert calls[0] <= 7


def test_golden_section_int_bounds_stay_in_range():
    f = lambda x: (x - 42) ** 2
    best_x, _, history = golden_section_search(f, lo=0, hi=100, scale='int', n_evals=15)
    assert 0 <= best_x <= 100
    assert all(0 <= x <= 100 for x, _ in history)
    assert all(isinstance(x, int) for x, _ in history)


def test_nelder_mead_converges_on_2d_parabola():
    # f(x,y) = (x-3)^2 + (y-7)^2 on [0,10]^2
    f = lambda p: (p[0] - 3) ** 2 + (p[1] - 7) ** 2
    bounds = [(0, 10, 'int'), (0, 10, 'int')]
    best, best_y, history = nelder_mead(f, bounds, n_evals=50)
    assert abs(best[0] - 3) <= 1
    assert abs(best[1] - 7) <= 1


def test_nelder_mead_converges_on_anisotropic_quadratic():
    # Anisotropic quadratic — condition number ~10, minimum at (25, 60).
    # This reflects realistic tuning landscapes (smooth, unimodal, mildly
    # anisotropic) better than Rosenbrock's curved valley.
    def f(p):
        return (p[0] - 25) ** 2 + 10 * (p[1] - 60) ** 2

    bounds = [(0, 100, 'int'), (0, 100, 'int')]
    best, best_y, history = nelder_mead(f, bounds, n_evals=100)
    assert abs(best[0] - 25) <= 5
    assert abs(best[1] - 60) <= 5


def test_nelder_mead_respects_log_scale():
    # Minimum at (log) 5000 and 50 — search in log space [100, 50000] x [10, 500]
    def f(p):
        return (math.log(max(p[0], 1)) - math.log(5000)) ** 2 + \
               (math.log(max(p[1], 1)) - math.log(50)) ** 2
    bounds = [(100, 50000, 'log'), (10, 500, 'log')]
    best, best_y, history = nelder_mead(f, bounds, n_evals=80)
    # Log-space convergence: within 50% of each target
    assert 2500 <= best[0] <= 10000
    assert 25 <= best[1] <= 100


def test_nelder_mead_respects_budget():
    calls = [0]
    def f(p):
        calls[0] += 1
        return sum(x * x for x in p)
    bounds = [(0, 10, 'int'), (0, 10, 'int'), (0, 10, 'int')]
    nelder_mead(f, bounds, n_evals=30)
    assert calls[0] <= 30  # may be slightly less due to cache dedup


def test_nelder_mead_stays_within_bounds():
    f = lambda p: sum(x for x in p)  # minimized at lower bound
    bounds = [(0, 10, 'int'), (5, 15, 'int')]
    best, _, history = nelder_mead(f, bounds, n_evals=40)
    for x, y in history:
        assert 0 <= x[0] <= 10
        assert 5 <= x[1] <= 15


def test_optimize_params_picks_golden_for_1d_algos():
    # sa has only 1 param: sa_iters
    calls = []
    def eval_fn(params):
        calls.append(params)
        return (params['sa_iters'] - 12000) ** 2
    best_params, best_score, history = optimize_params('sa', eval_fn, n_evals=12)
    assert 'sa_iters' in best_params
    assert abs(best_params['sa_iters'] - 12000) < 3000
    assert len(calls) <= 12


def test_optimize_params_picks_nelder_mead_for_2d_algos():
    # woa has 2 params: woa_pop (int) and woa_iters (log)
    def eval_fn(params):
        return (params['woa_pop'] - 40) ** 2 + (params['woa_iters'] - 500) ** 2 / 1000
    best_params, best_score, history = optimize_params('woa', eval_fn, n_evals=40)
    assert abs(best_params['woa_pop'] - 40) <= 10
    assert 200 <= best_params['woa_iters'] <= 5000


def test_optimize_params_returns_input_dict_shape():
    def eval_fn(params):
        return sum(v for v in params.values() if isinstance(v, (int, float)))
    best_params, _, _ = optimize_params('tabu', eval_fn, n_evals=20)
    # tabu has 3 params — dispatcher should return all 3
    assert set(best_params.keys()) == {'tabu_iters', 'tabu_tenure', 'tabu_patience'}


def test_optimize_params_raises_for_unknown_algo():
    with pytest.raises(KeyError):
        optimize_params('does_not_exist', lambda p: 0.0, n_evals=10)
