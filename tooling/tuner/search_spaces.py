"""Search spaces and defaults for the auto-tuner.

These are the knobs the tuner is allowed to touch, plus the seed set we
re-use across every evaluation so score differences come from parameters
not seed luck.

Adding a new tunable algo? Drop a ``(lo, hi, scale)`` tuple per knob into
:data:`SEARCH_SPACES` — everything downstream (sampling, perturbation,
optimizer) picks it up automatically.
"""
from __future__ import annotations

from tooling.tuned_params import load_params as _load_golden, FALLBACK_PARAMS  # noqa: F401


# (min, max, scale)  scale: 'log' = log-uniform, 'int' = uniform int
SEARCH_SPACES = {
    'tabu': {
        'tabu_iters':    (500,  20000, 'log'),
        'tabu_tenure':   (5,    50,    'int'),
        'tabu_patience': (50,   2000,  'log'),
    },
    'sa':    {'sa_iters':    (1000, 50000, 'log')},
    'kempe': {'kempe_iters': (500,  20000, 'log')},
    'alns':  {'alns_iters':  (500,  20000, 'log')},
    'gd':    {'gd_iters':    (1000, 50000, 'log')},
    'abc': {
        'abc_pop':   (10,  100,  'int'),
        'abc_iters': (500,  20000, 'log'),
    },
    'ga': {
        'ga_pop':   (20,  200,  'int'),
        'ga_iters': (100, 5000, 'log'),
    },
    'lahc': {
        'lahc_iters': (1000, 50000, 'log'),
        'lahc_list':  (0,    5000,  'int'),
    },
    'woa': {
        'woa_pop':   (10,  100,  'int'),
        'woa_iters': (500, 20000, 'log'),
    },
    'vns': {
        'vns_iters':  (1000, 50000, 'log'),
        'vns_budget': (10,   100,   'int'),
    },
}

DEFAULT_PARAMS = _load_golden()

TUNABLE_ALGOS = list(SEARCH_SPACES.keys())

# Fixed seed set for fair comparison — every config is tested on the same
# seeds so score differences reflect param quality, not seed luck.
EVAL_SEEDS = [42, 123, 789]
