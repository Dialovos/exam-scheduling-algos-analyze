#!/usr/bin/env python3
"""
Phase 4a — Bayesian hyperparameter tuning via Optuna.

Tunes per-algo hyperparameters (tabu_tenure, sa_cooling, etc.) on a given
ITC 2007 instance, using the C++ solver as a black-box evaluator.
Output: JSON with best params per (algo, instance) pair.

Usage:
    pip install optuna
    python3 tooling/bo_tune.py --instance exam_comp_set4 --algo tabu_cached --trials 50
    python3 tooling/bo_tune.py --all    # every algo × every ITC set (long!)

Typical trial budget: 30-60 trials per (algo, instance), ~15-60 min.
Optuna uses TPE (Tree-structured Parzen Estimator) — usually converges
on competitive params within 20-30 trials.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("error: pip install optuna")


ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cpp" / "build" / "exam_solver"
INSTANCE_DIR = ROOT / "instances"

# Per-algo parameter spaces. Extend as needed.
PARAM_SPACES = {
    "tabu_cached": {
        "tabu-iters":    lambda t: t.suggest_int("tabu_iters", 1000, 5000, step=500),
        "tabu-tenure":   lambda t: t.suggest_int("tabu_tenure", 10, 60, step=5),
        "tabu-patience": lambda t: t.suggest_int("tabu_patience", 300, 1200, step=100),
    },
    "sa_cached": {
        "sa-iters": lambda t: t.suggest_int("sa_iters", 30000, 120000, step=10000),
        # cooling-rate is hardcoded; exposing it would require a new CLI flag.
        # Document as a follow-up; for now tune iters only.
    },
    "gd_cached": {
        "gd-iters": lambda t: t.suggest_int("gd_iters", 30000, 120000, step=10000),
    },
    "lahc_cached": {
        "lahc-iters": lambda t: t.suggest_int("lahc_iters", 30000, 120000, step=10000),
        "lahc-list":  lambda t: t.suggest_int("lahc_list", 0, 0),  # 0 = auto
    },
    "alns_thompson": {
        "alns-iters": lambda t: t.suggest_int("alns_iters", 1000, 8000, step=500),
    },
}


def run_solver(instance: str, algo: str, params: dict, seed: int = 42) -> dict:
    """Run C++ solver with given params, return parsed JSON result."""
    cmd = [
        str(BIN),
        str(INSTANCE_DIR / f"{instance}.exam"),
        "--algo", algo,
        "--seed", str(seed),
    ]
    for flag, val in params.items():
        cmd += [f"--{flag}", str(val)]
    try:
        out = subprocess.check_output(cmd, timeout=600, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"feasible": False, "hard_violations": 999, "soft_penalty": 10**9}
    except subprocess.CalledProcessError:
        return {"feasible": False, "hard_violations": 999, "soft_penalty": 10**9}

    try:
        data = json.loads(out.decode())
        if isinstance(data, list) and data:
            data = data[0]
        return data
    except json.JSONDecodeError:
        return {"feasible": False, "hard_violations": 999, "soft_penalty": 10**9}


def objective_factory(instance: str, algo: str, n_seeds: int):
    """Closure: Optuna objective over 3-seed mean soft; feasibility-first."""
    space = PARAM_SPACES[algo]

    def objective(trial):
        params = {flag: fn(trial) for flag, fn in space.items()}
        softs, hards = [], []
        for seed in range(42, 42 + n_seeds):
            r = run_solver(instance, algo, params, seed)
            softs.append(int(r.get("soft_penalty", 10**9)))
            hards.append(int(r.get("hard_violations", 999)))
        # Feasibility-first: penalize hard heavily
        mean_hard = sum(hards) / len(hards)
        mean_soft = sum(softs) / len(softs)
        return mean_hard * 100000 + mean_soft

    return objective


def tune_one(instance: str, algo: str, trials: int, n_seeds: int, out_path: Path):
    print(f"Tuning {algo} on {instance} ({trials} trials, {n_seeds} seeds each)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(instance, algo, n_seeds),
                   n_trials=trials, show_progress_bar=True)
    best = study.best_params
    best_value = study.best_value
    print(f"  best value: {best_value}")
    print(f"  best params: {best}")

    # Merge into output JSON
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.setdefault(instance, {})[algo] = {
        "best_value": best_value,
        "params": best,
        "n_trials": trials,
        "n_seeds": n_seeds,
    }
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"  written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="exam_comp_set4")
    ap.add_argument("--algo", default="tabu_cached")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="seeds per trial (avg objective)")
    ap.add_argument("--all", action="store_true",
                    help="tune every algo × every ITC 2007 set")
    ap.add_argument("--out", default=str(ROOT / "tooling" / "tuned_params_v2.json"))
    args = ap.parse_args()

    if not BIN.exists():
        sys.exit(f"error: {BIN} missing. Run `make all` first.")

    out_path = Path(args.out)

    if args.all:
        algos = list(PARAM_SPACES.keys())
        sets = [f"exam_comp_set{i}" for i in range(1, 9)]
        for inst in sets:
            for algo in algos:
                tune_one(inst, algo, args.trials, args.seeds, out_path)
    else:
        if args.algo not in PARAM_SPACES:
            sys.exit(f"error: algo {args.algo} not in PARAM_SPACES. "
                     f"Supported: {list(PARAM_SPACES)}")
        tune_one(args.instance, args.algo, args.trials, args.seeds, out_path)


if __name__ == "__main__":
    main()
