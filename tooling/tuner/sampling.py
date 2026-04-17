"""Parameter sampling + chain mutation primitives.

Pure functions with an explicit RNG — calling these with the same RNG
state produces the same outputs, which keeps tuning runs reproducible.

Chain length is always clamped to ``[MIN_CHAIN_LEN, MAX_CHAIN_LEN]`` and
``allow_duplicates=False`` enforces that consecutive steps use different
algos (an ``sa→sa`` pair is almost never useful: the second SA sees the
first's final state and mostly re-explores it).
"""
from __future__ import annotations

import math

from tooling.tuner.search_spaces import (
    SEARCH_SPACES, DEFAULT_PARAMS,
    MAX_CHAIN_LEN, MIN_CHAIN_LEN, CHAIN_CROSSOVER_RATE,
)


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


def _pick_non_adjacent(top_algos, prev_algo, rng):
    """Return an algo from *top_algos* that differs from *prev_algo*.

    Falls back to any choice if the pool is size 1 (no alternatives).
    """
    if prev_algo is None or len(top_algos) <= 1:
        return rng.choice(top_algos)
    choices = [a for a in top_algos if a != prev_algo]
    return rng.choice(choices) if choices else rng.choice(top_algos)


def random_chain(top_algos, best_params, rng, length=None, allow_duplicates=True):
    """Build a random chain, each step perturbed around best known params.

    * ``length=None`` samples uniformly from ``[MIN_CHAIN_LEN, min(5, MAX_CHAIN_LEN)]``.
    * Callers can request a longer chain by passing ``length`` explicitly
      — it is clamped to ``MAX_CHAIN_LEN`` regardless.
    * When ``allow_duplicates=False`` adjacent identical algos are avoided
      (sa→sa offers no benefit over a single longer sa).
    """
    if length is None:
        length = rng.randint(MIN_CHAIN_LEN, min(5, MAX_CHAIN_LEN, len(top_algos)))
    length = max(MIN_CHAIN_LEN, min(MAX_CHAIN_LEN, length))
    chain = []
    prev_algo = None
    for _ in range(length):
        if allow_duplicates:
            algo = rng.choice(top_algos)
        else:
            algo = _pick_non_adjacent(top_algos, prev_algo, rng)
        params = perturb(algo, best_params.get(algo, DEFAULT_PARAMS.get(algo, {})),
                         rng, intensity=0.2)
        chain.append((algo, params))
        prev_algo = algo
    return chain


def mutate_chain(chain, top_algos, best_params, rng, allow_duplicates=True):
    """Apply one of {swap, perturb, add, remove} to a chain. GA-style variation.

    Respects ``MAX_CHAIN_LEN`` for the add op and ``MIN_CHAIN_LEN`` for remove.
    When ``allow_duplicates=False`` swap/add pick algos that don't match
    either neighbour.
    """
    chain = list(chain)
    op = rng.choice(['swap', 'perturb', 'add', 'remove'])
    if op == 'swap' and chain:
        i = rng.randrange(len(chain))
        forbidden = set()
        if not allow_duplicates:
            if i > 0:
                forbidden.add(chain[i - 1][0])
            if i + 1 < len(chain):
                forbidden.add(chain[i + 1][0])
        pool = [a for a in top_algos if a not in forbidden] or top_algos
        a = rng.choice(pool)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain[i] = (a, p)
    elif op == 'perturb' and chain:
        i = rng.randrange(len(chain))
        a, p = chain[i]
        chain[i] = (a, perturb(a, p, rng))
    elif op == 'add' and len(chain) < MAX_CHAIN_LEN:
        pos = rng.randint(0, len(chain))
        forbidden = set()
        if not allow_duplicates:
            if pos > 0:
                forbidden.add(chain[pos - 1][0])
            if pos < len(chain):
                forbidden.add(chain[pos][0])
        pool = [a for a in top_algos if a not in forbidden] or top_algos
        a = rng.choice(pool)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain.insert(pos, (a, p))
    elif op == 'remove' and len(chain) > MIN_CHAIN_LEN:
        chain.pop(rng.randrange(len(chain)))
    return chain


def crossover(parent_a, parent_b, rng, allow_duplicates=True):
    """1-point crossover: prefix of *a* + suffix of *b*.

    Cut points are independent on each parent so the child length spans
    ``[MIN_CHAIN_LEN, MAX_CHAIN_LEN]``. Pure recombination — it does *not*
    perturb params (use :func:`mutate_chain` for that).

    When ``allow_duplicates=False`` and the prefix's last algo matches the
    suffix's first algo, the duplicate step is dropped from the suffix so
    the join point is clean. The result is clamped to ``[MIN_CHAIN_LEN,
    MAX_CHAIN_LEN]``.
    """
    if not parent_a or not parent_b:
        return list(parent_a or parent_b)
    cut_a = rng.randint(1, len(parent_a))
    cut_b = rng.randint(0, len(parent_b) - 1)
    prefix = list(parent_a[:cut_a])
    suffix = list(parent_b[cut_b:])
    if not allow_duplicates and prefix and suffix and prefix[-1][0] == suffix[0][0]:
        suffix = suffix[1:]
    child = prefix + suffix
    if not allow_duplicates:
        deduped = []
        prev = None
        for step in child:
            if step[0] != prev:
                deduped.append(step)
                prev = step[0]
        child = deduped
    if len(child) > MAX_CHAIN_LEN:
        child = child[:MAX_CHAIN_LEN]
    if len(child) < MIN_CHAIN_LEN:
        # Pad from the longer parent's remaining tail
        donor = parent_a if len(parent_a) >= len(parent_b) else parent_b
        for step in donor:
            if len(child) >= MIN_CHAIN_LEN:
                break
            if allow_duplicates or not child or child[-1][0] != step[0]:
                child.append(step)
    return child


def vary_chain(parent, top_algos, best_params, rng, survivors=None,
               crossover_rate=CHAIN_CROSSOVER_RATE, allow_duplicates=True):
    """Apply either crossover (with probability *crossover_rate*) or mutation.

    When crossover is chosen but *survivors* has < 2 entries, falls back to
    mutation — crossover needs a second parent. Caller passes the current
    survivor pool so the second parent is drawn from known-good chains.
    """
    use_crossover = (
        survivors is not None
        and len(survivors) >= 2
        and rng.random() < crossover_rate
    )
    if use_crossover:
        others = [s for s in survivors if s is not parent] or survivors
        other = rng.choice(others)
        return crossover(parent, other, rng, allow_duplicates=allow_duplicates)
    return mutate_chain(parent, top_algos, best_params, rng,
                        allow_duplicates=allow_duplicates)
