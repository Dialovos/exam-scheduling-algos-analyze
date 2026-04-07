"""
Tuned Parameters — Single source of truth for algorithm defaults.

Reads from tuned_params.json if it exists, otherwise falls back to
hardcoded defaults. Maintains a version log for rollback.

Usage:
    from tuned_params import load_params, save_params, get

    params = load_params()           # full dict: {algo: {param: val, ...}, ...}
    sa_params = get('sa')            # {'sa_iters': 5000}
    flat = load_params_flat()        # flattened: {'sa_iters': 5000, 'tabu_iters': 2000, ...}
"""

import json
import os
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent
_PARAMS_FILE = _ROOT / 'tuned_params.json'
_LOG_FILE = _ROOT / 'tuned_params_log.json'

# ── Hardcoded fallback (last-resort defaults) ───────────────

FALLBACK_PARAMS = {
    'tabu':  {'tabu_iters': 2000, 'tabu_tenure': 20, 'tabu_patience': 500},
    'hho':   {'hho_pop': 30,  'hho_iters': 200},
    'sa':    {'sa_iters': 5000},
    'kempe': {'kempe_iters': 3000},
    'alns':  {'alns_iters': 2000},
    'gd':    {'gd_iters': 5000},
    'abc':   {'abc_pop': 30,  'abc_iters': 3000},
    'ga':    {'ga_pop': 50,   'ga_iters': 500},
    'lahc':  {'lahc_iters': 5000, 'lahc_list': 0},
}


# ── Load / Get ──────────────────────────────────────────────

def load_params():
    """Load active params dict. Falls back to FALLBACK_PARAMS on any error."""
    try:
        if _PARAMS_FILE.is_file():
            with open(_PARAMS_FILE) as f:
                data = json.load(f)
            return data.get('params', FALLBACK_PARAMS)
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return dict(FALLBACK_PARAMS)


def get(algo):
    """Get params for a specific algorithm."""
    return load_params().get(algo, {})


def load_params_flat():
    """Load params flattened into a single dict (for argparse defaults / cpp_bridge)."""
    flat = {}
    for algo_params in load_params().values():
        flat.update(algo_params)
    return flat


def load_metadata():
    """Load the full tuned_params.json including metadata. Returns None if missing."""
    try:
        if _PARAMS_FILE.is_file():
            with open(_PARAMS_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


# ── Save ────────────────────────────────────────────────────

def save_params(params, aggregate_score, per_dataset_scores=None,
                trial_count=0, source='auto_tuner'):
    """Save new params atomically + append to version log.

    Args:
        params: {algo: {param: val, ...}, ...}
        aggregate_score: geometric mean or raw score
        per_dataset_scores: {dataset_label: score, ...} or None
        trial_count: how many trials produced this result
        source: 'auto_tuner' or 'manual_rollback'
    """
    version = _next_version()

    record = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'aggregate_score': aggregate_score,
        'per_dataset_scores': per_dataset_scores or {},
        'trial_count': trial_count,
        'source': source,
    }

    # Atomic write to active params file
    tmp = str(_PARAMS_FILE) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, _PARAMS_FILE)

    # Append to version log
    _append_log(record)

    return version


def _next_version():
    """Get next version number from log."""
    log = _load_log()
    if not log:
        return 1
    return max(entry.get('version', 0) for entry in log) + 1


def _append_log(record):
    """Append a record to the version log (atomic)."""
    log = _load_log()
    log.append(record)
    tmp = str(_LOG_FILE) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, _LOG_FILE)


def _load_log():
    """Load the version log. Returns [] on error."""
    try:
        if _LOG_FILE.is_file():
            with open(_LOG_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return []


# ── Rollback ────────────────────────────────────────────────

def rollback(version):
    """Restore params from a specific version in the log.

    Returns the restored version number, or None if not found.
    """
    log = _load_log()
    for entry in log:
        if entry.get('version') == version:
            save_params(
                params=entry['params'],
                aggregate_score=entry.get('aggregate_score', 0),
                per_dataset_scores=entry.get('per_dataset_scores', {}),
                trial_count=entry.get('trial_count', 0),
                source='manual_rollback',
            )
            return version
    return None


def list_versions():
    """Return list of (version, timestamp, score, source) tuples."""
    log = _load_log()
    return [
        (e.get('version', '?'),
         e.get('timestamp', '?'),
         e.get('aggregate_score', '?'),
         e.get('source', '?'))
        for e in log
    ]


# ── Comparison helpers ──────────────────────────────────────

def check_should_update(new_score, new_per_dataset, new_trial_count,
                        max_regression=0.15):
    """Check if new params should replace current ones.

    Returns (should_update: bool, reason: str).

    Rules:
      1. New aggregate score must be better (lower)
      2. Trial counts must be comparable (within 50%)
      3. No single dataset regressed > max_regression (15%)
    """
    current = load_metadata()

    # No existing params — always accept first tuning result
    if current is None:
        return True, "no existing tuned params, accepting first result"

    old_score = current.get('aggregate_score', float('inf'))
    old_per_ds = current.get('per_dataset_scores', {})
    old_trials = current.get('trial_count', 0)

    # Gate 1: aggregate must improve
    if new_score >= old_score:
        return False, f"no improvement: new={new_score:.4f} >= old={old_score:.4f}"

    # Gate 2: trial counts comparable (skip if old has no count)
    if old_trials > 0 and new_trial_count > 0:
        ratio = new_trial_count / old_trials
        if ratio < 0.5 or ratio > 2.0:
            return False, (f"trial count mismatch: new={new_trial_count} "
                           f"vs old={old_trials} (ratio={ratio:.2f})")

    # Gate 3: no catastrophic per-dataset regression
    if old_per_ds and new_per_dataset:
        for ds, new_val in new_per_dataset.items():
            old_val = old_per_ds.get(ds)
            if old_val is not None and old_val > 0:
                regression = (new_val - old_val) / old_val
                if regression > max_regression:
                    return False, (f"dataset '{ds}' regressed {regression:.1%}: "
                                   f"new={new_val:.0f} vs old={old_val:.0f}")

    improvement = (old_score - new_score) / old_score if old_score > 0 else 1.0
    return True, f"improved {improvement:.1%}: new={new_score:.4f} vs old={old_score:.4f}"


def check_plateau(min_improvement=0.01, window=3):
    """Check if recent updates have plateaued.

    Returns (is_plateau: bool, reason: str).
    Plateau = last `window` versions improved aggregate < min_improvement total.
    """
    log = _load_log()
    if len(log) < window + 1:
        return False, f"only {len(log)} versions, need {window + 1} to check plateau"

    recent = log[-window:]
    older = log[-(window + 1)]

    older_score = older.get('aggregate_score', float('inf'))
    best_recent = min(e.get('aggregate_score', float('inf')) for e in recent)

    if older_score <= 0 or older_score == float('inf'):
        return False, "cannot compute plateau (bad older score)"

    total_improvement = (older_score - best_recent) / older_score
    if total_improvement < min_improvement:
        return True, (f"plateau: last {window} versions improved only "
                      f"{total_improvement:.2%} (threshold: {min_improvement:.0%})")

    return False, f"still improving: {total_improvement:.2%} over last {window} versions"
