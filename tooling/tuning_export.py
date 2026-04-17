#!/usr/bin/env python3
"""Convert ``tuned_params_log.json`` → tidy long-form CSV.

``tuned_params_log.json`` records every save of the golden-params file,
one entry per version. Each entry has the full per-algo param snapshot,
aggregate score, per-dataset scores, and source metadata.

Analysts usually want these in a relational format so they can pivot/
filter with pandas. We emit two tidy CSVs:

    * ``<out>_algos.csv``    — one row per (version, algorithm, param)
    * ``<out>_datasets.csv`` — one row per (version, dataset)

If ``--out`` ends in ``.csv`` we use it as-is for the algos file and
derive the datasets filename by swapping the stem suffix. If it's a bare
stem we generate both with canonical suffixes.

Usage:
    python -m tooling.tuning_export tooling/tuned_params_log.json
    python -m tooling.tuning_export tooling/tuned_params_log.json --out /tmp/tx.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _sibling(out_path: Path, suffix: str) -> Path:
    """``/tmp/tx.csv``, ``_algos`` → ``/tmp/tx_algos.csv``."""
    return out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix or '.csv'}")


def export(log_path: Path, out_path: Path) -> tuple[Path, Path]:
    """Write both CSVs and return ``(algos_path, datasets_path)``."""
    log = json.loads(log_path.read_text())
    if not isinstance(log, list):
        raise SystemExit(f"Expected a JSON list, got {type(log).__name__}")

    algos_path = _sibling(out_path, "_algos") if out_path.suffix else out_path.with_suffix(".csv")
    ds_path = _sibling(out_path, "_datasets") if out_path.suffix else algos_path.with_name(f"{algos_path.stem}_datasets.csv")

    algo_rows = []
    ds_rows = []
    for entry in log:
        version = entry.get("version")
        timestamp = entry.get("timestamp", "")
        source = entry.get("source", "")
        agg = entry.get("aggregate_score")
        trials = entry.get("trial_count")
        params = entry.get("params", {}) or {}
        per_ds = entry.get("per_dataset_scores", {}) or {}

        for algo, knobs in params.items():
            if not isinstance(knobs, dict):
                continue
            for knob, value in knobs.items():
                algo_rows.append({
                    "version": version,
                    "timestamp": timestamp,
                    "source": source,
                    "algorithm": algo,
                    "param": knob,
                    "value": value,
                    "aggregate_score": agg,
                    "trial_count": trials,
                })

        for ds, score in per_ds.items():
            ds_rows.append({
                "version": version,
                "timestamp": timestamp,
                "source": source,
                "dataset": ds,
                "score": score,
                "aggregate_score": agg,
            })

    algos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(algos_path, "w", newline="") as f:
        if algo_rows:
            w = csv.DictWriter(f, fieldnames=list(algo_rows[0].keys()))
            w.writeheader()
            w.writerows(algo_rows)
    with open(ds_path, "w", newline="") as f:
        if ds_rows:
            w = csv.DictWriter(f, fieldnames=list(ds_rows[0].keys()))
            w.writeheader()
            w.writerows(ds_rows)

    print(f"wrote {algos_path} ({len(algo_rows)} rows)")
    print(f"wrote {ds_path} ({len(ds_rows)} rows)")
    return algos_path, ds_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="Path to tuned_params_log.json")
    ap.add_argument("--out", type=Path, default=Path("tuning_export.csv"),
                    help="Output CSV path (default: tuning_export.csv)")
    args = ap.parse_args()
    if not args.log.is_file():
        raise SystemExit(f"Log not found: {args.log}")
    export(args.log, args.out)


if __name__ == "__main__":
    main()
