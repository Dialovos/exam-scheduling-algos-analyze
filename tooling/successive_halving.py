"""Successive Halving for chain/config evaluation with budget allocation.

Standard SH: start with N candidates at low fidelity (1 seed, 1 dataset).
Promote top 1/eta to higher fidelity. Repeat until 1 (or k) survivors.

Rungs are specified as [(n_seeds_r0, n_datasets_r0), (n_seeds_r1, n_datasets_r1), ...]
where each later rung has higher fidelity. Between rungs, the top 1/eta of
candidates are promoted (default eta=2, i.e. keep best half).

The caller supplies eval_fn(candidate, rung_idx, fidelity) -> float, where
fidelity is the tuple (n_seeds, n_datasets) at that rung. eval_fn is
responsible for honoring the fidelity (running more seeds/datasets when asked).

Lower scores are better. Ties broken by stable sort (first candidate wins).
"""
from concurrent.futures import ThreadPoolExecutor


def successive_halving(candidates, eval_fn, rungs, eta=2, eta_schedule=None):
    """Run successive halving.

    Args:
        candidates: list of candidate objects (chains, configs — anything).
        eval_fn: callable(candidate, rung_idx, fidelity_tuple) -> float.
        rungs: list of fidelity tuples, innermost first. Length = number
               of rungs. Rung k uses fidelity rungs[k].
        eta: reduction factor per rung (default 2, i.e. halving).
        eta_schedule: optional per-rung reduction factor list. When provided,
               rung ``k`` keeps top ``len(scored) // eta_schedule[k]``. Must be
               ``len(rungs) - 1`` long (no promotion after final rung — the
               final entry is unused and may be omitted). Falls back to *eta*
               when None. Lets large populations prune aggressively at rung 0
               (e.g., eta=4) while small ones stay conservative (eta=2).

    Returns:
        (winner_candidate, winner_score, history)
        history: list of (candidate, rung_idx, score)
    """
    if not candidates:
        raise ValueError("successive_halving: empty candidate list")
    if not rungs:
        raise ValueError("successive_halving: empty rungs list")

    # Track best score ever seen per candidate (across rungs) so we can
    # return meaningful history even if later rungs return inf.
    survivors = list(candidates)
    scores = {}  # id(candidate) -> (candidate, best_score, last_rung)
    history = []

    for rung_idx, fidelity in enumerate(rungs):
        # Evaluate all survivors in parallel — each eval_fn call launches
        # subprocesses internally, so threads overlap the I/O wait.
        def _eval(c, _ri=rung_idx, _f=fidelity):
            return c, _ri, eval_fn(c, _ri, _f)

        rung_scores = []
        with ThreadPoolExecutor(max_workers=max(1, len(survivors))) as pool:
            for c, ri, s in pool.map(_eval, survivors):
                history.append((c, ri, s))
                rung_scores.append((c, s))
                key = id(c)
                if key not in scores or s < scores[key][1]:
                    scores[key] = (c, s, ri)

        if rung_idx == len(rungs) - 1:
            break  # no promotion after final rung

        # Promote top 1/eta (per-rung override via eta_schedule)
        rung_scores.sort(key=lambda p: p[1])
        rung_eta = eta_schedule[rung_idx] if eta_schedule and rung_idx < len(eta_schedule) else eta
        n_promote = max(1, len(rung_scores) // rung_eta)
        survivors = [c for c, _ in rung_scores[:n_promote]]

    # Winner = candidate with lowest best score across all its rungs
    best_entry = min(scores.values(), key=lambda e: e[1])
    return best_entry[0], best_entry[1], history
