"""Derivative-free optimizers for auto-tuner parameter search.

Two routines, both stdlib-only:

* golden_section_search — 1D unimodal minimization, log or int scale.
* nelder_mead            — n-D simplex-based minimization, bounded, log/int scale.

Both routines are deterministic given the same budget (no randomness inside
the algorithms themselves — randomness lives in the fitness function, e.g.
solver seeds). They return (best_x, best_y, history) where history is a list
of (x, y) tuples in evaluation order. The fitness function f must be
non-None and return a finite float (infeasible configurations should return
a large penalty like 1e9, not NaN).
"""
import math


_PHI = (1 + math.sqrt(5)) / 2  # golden ratio
_INV_PHI = 1 / _PHI
_INV_PHI_SQ = 1 / (_PHI * _PHI)


def _project_int(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


def golden_section_search(f, lo, hi, scale='int', n_evals=15):
    """Golden-section search for unimodal minimization on [lo, hi].

    Args:
        f: callable (x) -> float. x is int for scale='int' or 'log'.
        lo, hi: inclusive bounds (int).
        scale: 'int' (uniform) or 'log' (log-uniform — search in log space,
               callback gets rounded int).
        n_evals: total function evaluations allowed (>=3).

    Returns:
        (best_x, best_y, history)
        history: list of (x, y) in call order.
    """
    if n_evals < 3:
        raise ValueError("golden_section_search needs n_evals >= 3")

    # Work in transformed space
    if scale == 'log':
        a, b = math.log(max(lo, 1)), math.log(max(hi, 1))
        def to_native(t): return _project_int(math.exp(t), lo, hi)
    else:
        a, b = float(lo), float(hi)
        def to_native(t): return _project_int(t, lo, hi)

    history = []
    cache = {}

    def eval_native(x_native):
        if x_native in cache:
            return cache[x_native]
        y = f(x_native)
        cache[x_native] = y
        history.append((x_native, y))
        return y

    # Initial two interior points
    c = b - (b - a) * _INV_PHI
    d = a + (b - a) * _INV_PHI
    fc = eval_native(to_native(c))
    fd = eval_native(to_native(d))

    # Each iteration does ONE new eval (except the first two above).
    # So we run n_evals - 2 iterations.
    for _ in range(n_evals - 2):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * _INV_PHI
            fc = eval_native(to_native(c))
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * _INV_PHI
            fd = eval_native(to_native(d))

    # Final: pick best seen (not just last interval midpoint — cache dedup
    # may have squeezed the evaluation count below n_evals for pathological
    # unimodal inputs, but history always reflects what was tried).
    best_x, best_y = min(history, key=lambda p: p[1])
    # Pad history to exactly n_evals by repeating the best point (only if
    # the cache shortcut fired — keeps the contract "history has n_evals").
    while len(history) < n_evals:
        history.append((best_x, best_y))
    return best_x, best_y, history


def _transform_forward(raw_vec, bounds):
    """Map raw int values -> normalized unit-space [0, 1]^n."""
    out = []
    for v, (lo, hi, scale) in zip(raw_vec, bounds):
        if scale == 'log':
            lg_lo, lg_hi = math.log(max(lo, 1)), math.log(max(hi, 1))
            lg = math.log(max(v, 1))
            out.append((lg - lg_lo) / (lg_hi - lg_lo) if lg_hi > lg_lo else 0.5)
        else:
            out.append((v - lo) / (hi - lo) if hi > lo else 0.5)
    return out


def _transform_back(unit_vec, bounds):
    """Map normalized [0,1]^n -> raw int values (respecting scale)."""
    out = []
    for u, (lo, hi, scale) in zip(unit_vec, bounds):
        u = max(0.0, min(1.0, u))
        if scale == 'log':
            lg_lo, lg_hi = math.log(max(lo, 1)), math.log(max(hi, 1))
            v = math.exp(lg_lo + u * (lg_hi - lg_lo))
        else:
            v = lo + u * (hi - lo)
        out.append(_project_int(v, lo, hi))
    return tuple(out)


def nelder_mead(f, bounds, n_evals=60, initial=None):
    """Nelder-Mead simplex minimization in normalized unit-space.

    Args:
        f: callable (tuple[int, ...]) -> float.
        bounds: list of (lo, hi, scale) — scale in {'int', 'log'}.
        n_evals: total function evaluations allowed (>= n+1 where n=dim).
        initial: optional seed raw point (tuple of ints). If None, uses
                 center of bounds.

    Returns:
        (best_raw, best_y, history)
        history: list of (raw_tuple, y) in call order.
    """
    n = len(bounds)
    if n_evals < n + 2:
        raise ValueError(f"nelder_mead needs n_evals >= {n + 2}")

    # Build initial simplex in unit space (n+1 vertices).
    if initial is None:
        x0_unit = [0.5] * n
    else:
        x0_unit = _transform_forward(initial, bounds)

    simplex = [list(x0_unit)]
    for i in range(n):
        v = list(x0_unit)
        v[i] = 0.2 if x0_unit[i] > 0.5 else 0.8  # step from center
        simplex.append(v)

    history = []
    cache = {}
    # `iters` counts LOGICAL evaluations (including cache hits) to bound
    # the outer loop. Without this, the loop can spin forever once the
    # integer-rounded simplex collapses onto already-cached points —
    # `history` only grows on cache misses.
    iters = [0]

    def eval_unit(u):
        raw = _transform_back(u, bounds)
        iters[0] += 1
        if raw in cache:
            return cache[raw]
        y = f(raw)
        cache[raw] = y
        history.append((raw, y))
        return y

    fvals = [eval_unit(v) for v in simplex]

    # Nelder-Mead coefficients
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    def clip(u):
        return [max(0.0, min(1.0, x)) for x in u]

    while iters[0] < n_evals:
        # Order vertices by fvals
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        best, worst, second_worst = order[0], order[-1], order[-2]

        # Centroid of all but the worst
        centroid = [0.0] * n
        for i in order[:-1]:
            for j in range(n):
                centroid[j] += simplex[i][j]
        centroid = [c / n for c in centroid]

        # Reflection
        xr = clip([centroid[j] + alpha * (centroid[j] - simplex[worst][j]) for j in range(n)])
        fr = eval_unit(xr)
        if iters[0] >= n_evals:
            break

        if fvals[best] <= fr < fvals[second_worst]:
            simplex[worst], fvals[worst] = xr, fr
            continue

        # Expansion
        if fr < fvals[best]:
            xe = clip([centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)])
            fe = eval_unit(xe)
            if len(history) >= n_evals:
                break
            if fe < fr:
                simplex[worst], fvals[worst] = xe, fe
            else:
                simplex[worst], fvals[worst] = xr, fr
            continue

        # Contraction
        xc = clip([centroid[j] + rho * (simplex[worst][j] - centroid[j]) for j in range(n)])
        fc = eval_unit(xc)
        if iters[0] >= n_evals:
            break
        if fc < fvals[worst]:
            simplex[worst], fvals[worst] = xc, fc
            continue

        # Shrink toward best
        for i in range(n + 1):
            if i == best:
                continue
            simplex[i] = clip([simplex[best][j] + sigma * (simplex[i][j] - simplex[best][j]) for j in range(n)])
            fvals[i] = eval_unit(simplex[i])
            if len(history) >= n_evals:
                break

    best_raw, best_y = min(history, key=lambda p: p[1])
    return best_raw, best_y, history


def optimize_params(algo, eval_fn, n_evals, search_spaces=None):
    """Unified parameter optimizer dispatcher.

    Picks golden_section_search for 1D algos and nelder_mead for 2D+.

    Args:
        algo: algo name (key into search_spaces).
        eval_fn: callable (params_dict) -> float (lower is better).
        n_evals: budget of function evaluations.
        search_spaces: dict of algo -> {param_name: (lo, hi, scale)}.
                       If None, imports from tooling.auto_tuner.

    Returns:
        (best_params_dict, best_score, history_of_(params_dict, score))
    """
    if search_spaces is None:
        from tooling.auto_tuner import SEARCH_SPACES
        search_spaces = SEARCH_SPACES

    if algo not in search_spaces:
        raise KeyError(f"Unknown algo '{algo}' — not in search_spaces")

    space = search_spaces[algo]
    param_names = list(space.keys())

    if len(param_names) == 1:
        name = param_names[0]
        lo, hi, scale = space[name]

        def f(x):
            return eval_fn({name: x})

        best_x, best_y, hist = golden_section_search(f, lo, hi, scale, n_evals)
        history = [({name: x}, y) for x, y in hist]
        return {name: best_x}, best_y, history

    # 2D+: Nelder-Mead in normalized space
    bounds = [space[n] for n in param_names]

    def f(vec):
        params = {n: v for n, v in zip(param_names, vec)}
        return eval_fn(params)

    best_raw, best_y, hist = nelder_mead(f, bounds, n_evals)
    best_params = {n: v for n, v in zip(param_names, best_raw)}
    history = [({n: v for n, v in zip(param_names, vec)}, y) for vec, y in hist]
    return best_params, best_y, history
