"""Parameter sampling + chain mutation primitives.

Pure functions with an explicit RNG — calling these with the same RNG
state produces the same outputs, which keeps tuning runs reproducible.
"""
from __future__ import annotations

import math

from tooling.tuner.search_spaces import SEARCH_SPACES, DEFAULT_PARAMS


def _sample_val(lo, hi, scale, rng):
    if scale == 'log':
        return int(round(math.exp(rng.uniform(math.log(max(lo, 1)),
                                               math.log(max(hi, 1))))))
    return rng.randint(lo, hi)


def sample_random(algo, rng):
    return {k: _sample_val(*v, rng) for k, v in SEARCH_SPACES.get(algo, {}).items()}


def perturb(algo, base, rng, intensity=0.3):
    """Gaussian perturbation around *base* in the algo's search space.

    40% of knobs are kept as-is each call so we explore the neighbourhood
    rather than thrashing every dimension at once.
    """
    space = SEARCH_SPACES.get(algo, {})
    out = dict(base)
    for name, (lo, hi, scale) in space.items():
        if rng.random() < 0.4:
            continue
        bv = out.get(name, (lo + hi) // 2)
        if scale == 'log':
            log_lo, log_hi = math.log(max(lo, 1)), math.log(max(hi, 1))
            nv = math.exp(math.log(max(bv, 1)) + rng.gauss(0, intensity * (log_hi - log_lo)))
            out[name] = int(round(max(lo, min(hi, nv))))
        else:
            nv = bv + int(round(rng.gauss(0, intensity * (hi - lo))))
            out[name] = max(lo, min(hi, nv))
    return out


def random_chain(top_algos, best_params, rng, length=None):
    """Build a random chain of 2..4 algos, each perturbed around best known params."""
    if length is None:
        length = rng.randint(2, min(4, len(top_algos)))
    chain = []
    for _ in range(length):
        algo = rng.choice(top_algos)
        params = perturb(algo, best_params.get(algo, DEFAULT_PARAMS.get(algo, {})),
                         rng, intensity=0.2)
        chain.append((algo, params))
    return chain


def mutate_chain(chain, top_algos, best_params, rng):
    """Apply one of {swap, perturb, add, remove} to a chain. GA-style variation."""
    chain = list(chain)
    op = rng.choice(['swap', 'perturb', 'add', 'remove'])
    if op == 'swap' and chain:
        i = rng.randrange(len(chain))
        a = rng.choice(top_algos)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain[i] = (a, p)
    elif op == 'perturb' and chain:
        i = rng.randrange(len(chain))
        a, p = chain[i]
        chain[i] = (a, perturb(a, p, rng))
    elif op == 'add' and len(chain) < 5:
        a = rng.choice(top_algos)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain.insert(rng.randint(0, len(chain)), (a, p))
    elif op == 'remove' and len(chain) > 2:
        chain.pop(rng.randrange(len(chain)))
    return chain
