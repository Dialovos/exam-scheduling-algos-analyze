"""CLI entry point for the auto-tuner.

Run as:
    python -m tooling.auto_tuner instances/exam_comp_set4.exam
    python -m tooling.auto_tuner --all-sets --synthetic
    python -m tooling.auto_tuner --all-sets --resume
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tooling.tuner.core import AutoTuner
from tooling.tuner.synthetic import generate_synthetic_dataset


def main():
    ap = argparse.ArgumentParser(
        description="Auto-Tuner for Exam Scheduling Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tooling.auto_tuner instances/exam_comp_set4.exam            # single dataset
  python -m tooling.auto_tuner --all-sets                               # all ITC 2007 sets
  python -m tooling.auto_tuner --all-sets --synthetic                   # all + synthetic
  python -m tooling.auto_tuner instances/exam_comp_set1.exam instances/exam_comp_set4.exam  # specific sets
  python -m tooling.auto_tuner --all-sets --resume                      # resume from checkpoint
  python -m tooling.auto_tuner --all-sets --max-time 20                 # 20 min budget
        """)

    ap.add_argument('datasets', nargs='*', help='Dataset .exam files')
    ap.add_argument('--all-sets', action='store_true',
                    help='Use all ITC 2007 datasets (instances/exam_comp_set*.exam)')
    ap.add_argument('--synthetic', action='store_true',
                    help='Include a generated synthetic dataset')
    ap.add_argument('--output', default='tuning_results',
                    help='Output directory (default: tuning_results)')
    ap.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    ap.add_argument('--workers', type=int, default=6,
                    help='Max parallel workers (default: 6)')
    ap.add_argument('--param-trials', type=int, default=8,
                    help='Param trials per algo (default: 8)')
    ap.add_argument('--chain-pop', type=int, default=8,
                    help='Chain population (default: 8)')
    ap.add_argument('--chain-rounds', type=int, default=3,
                    help='Chain tournament rounds (default: 3)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    ap.add_argument('--eval-datasets', type=int, default=3,
                    help='Datasets to evaluate per trial in global mode (default: 3)')
    ap.add_argument('--max-time', type=int, default=None,
                    help='Wall-time budget in minutes (default: 30 for multi, 10 for single)')
    ap.add_argument('--ip-time-limit', type=int, default=120,
                    help='IP solver time limit seconds (default: 120)')
    ap.add_argument('--no-auto-update', action='store_true',
                    help='Do not auto-update golden params after tuning')
    ap.add_argument('--force-update', action='store_true',
                    help='Force-update golden params even if checks fail')

    args = ap.parse_args()

    # Resolve datasets
    datasets = list(args.datasets)
    if args.all_sets:
        # cli.py lives in tooling/tuner/, so instances/ is two levels up
        root = Path(__file__).resolve().parent.parent.parent / 'instances'
        itc_sets = sorted(root.glob('exam_comp_set*.exam'))
        datasets.extend(str(p) for p in itc_sets)

    if args.synthetic:
        syn_path = generate_synthetic_dataset(
            os.path.join(args.output, '_synthetic'), seed=args.seed)
        datasets.append(syn_path)

    # Deduplicate
    seen = set()
    unique = []
    for d in datasets:
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            unique.append(ad)
    datasets = unique

    if not datasets:
        print("Error: no datasets specified. Use positional args or --all-sets")
        sys.exit(1)

    for d in datasets:
        if not os.path.isfile(d):
            print(f"Error: dataset not found: {d}")
            sys.exit(1)

    max_time_sec = args.max_time * 60 if args.max_time else None

    AutoTuner(
        datasets=datasets,
        output_dir=args.output,
        max_workers=args.workers,
        param_trials=args.param_trials,
        chain_pop=args.chain_pop,
        chain_rounds=args.chain_rounds,
        seed=args.seed,
        ip_time_limit=args.ip_time_limit,
        max_time=max_time_sec,
        eval_datasets=args.eval_datasets,
        auto_update=not args.no_auto_update,
        force_update=args.force_update,
        resume=args.resume,
    ).run(resume=args.resume)


if __name__ == '__main__':
    main()
