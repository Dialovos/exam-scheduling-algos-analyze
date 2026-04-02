"""
Records every algorithm run with full metrics:
  - Runtime, feasibility, hard/soft violations
  - Full soft-constraint breakdown (7 components)
  - Memory usage, iterations, solver status
  - Dataset metadata (name, exams, periods, rooms, students, density)
  - Algorithm config (iterations, population, tenure, etc.)
  - Timestamp for tracking experiments

Storage: JSON-lines (one JSON object per line) for safe incremental writes.
Aggregation: computes mean/std/min/max across trials grouped by (algorithm, dataset).
Export: pandas DataFrame, CSV.

Usage:
    logger = ResultsLogger("results/run_log.jsonl")
    logger.log_run(dataset_name, problem, algo_result, config={...})
    df = logger.to_dataframe()
    agg = logger.aggregate()
"""

import json
import os
import time
import tracemalloc
from datetime import datetime
from collections import defaultdict


class ResultsLogger:
    """Append-only logger for algorithm runs."""

    def __init__(self, filepath: str = "results/run_log.jsonl"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    def log_run(
        self,
        dataset_name: str,
        problem,
        result: dict,
        config: dict = None,
        trial: int = 0,
        notes: str = "",
    ) -> dict:
        """Log a single algorithm run.

        Args:
            dataset_name: e.g. "exam_comp_set4"
            problem: ProblemInstance
            result: dict from solve_greedy/solve_tabu/etc with keys:
                     solution, runtime, evaluation, algorithm, iterations
            config: algorithm parameters used (e.g. {"tabu_iters": 2000})
            trial: trial number for repeated experiments
            notes: free-text annotation

        Returns:
            The logged record dict.
        """
        ev = result['evaluation']

        # Normalize attribute access (EvalBreakdown vs EvaluationResult)
        def _g(attr, fallback_attr=None):
            v = getattr(ev, attr, None)
            if v is not None:
                return v
            if fallback_attr:
                return getattr(ev, fallback_attr, 0)
            return 0

        record = {
            # ── Metadata ──
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "trial": trial,
            "notes": notes,

            # ── Dataset info ──
            "num_exams": problem.num_exams(),
            "num_periods": problem.num_periods(),
            "num_rooms": problem.num_rooms(),
            "num_students": problem.num_students(),
            "conflict_density": round(problem.conflict_density(), 4),
            "num_period_constraints": len(problem.period_hard_constraints),
            "num_room_constraints": len(problem.room_hard_constraints),

            # ── Algorithm ──
            "algorithm": result.get('algorithm', 'Unknown'),
            "config": config or {},

            # ── Performance ──
            "runtime": round(result.get('runtime', 0), 6),
            "iterations": result.get('iterations', 0),
            "memory_peak_mb": round(result.get('memory_peak_mb', 0), 2),
            "solver_status": result.get('solver_status', None),

            # ── Quality ──
            "feasible": bool(_g('feasible', 'is_feasible')),
            "hard_violations": int(_g('hard', 'hard_violations')),
            "soft_penalty": int(_g('soft', 'soft_penalty')),

            # ── Hard breakdown ──
            "conflicts": int(_g('conflicts')),
            "room_occupancy": int(_g('room_occupancy')),
            "period_utilisation": int(_g('period_utilisation')),
            "period_related": int(_g('period_related')),
            "room_related": int(_g('room_related')),

            # ── Soft breakdown ──
            "two_in_a_row": int(_g('two_in_a_row')),
            "two_in_a_day": int(_g('two_in_a_day')),
            "period_spread": int(_g('period_spread')),
            "non_mixed_durations": int(_g('non_mixed_durations')),
            "front_load": int(_g('front_load')),
            "period_penalty": int(_g('period_penalty')),
            "room_penalty": int(_g('room_penalty')),
        }

        # Append to file
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')

        return record

    def log_run_with_memory(
        self,
        dataset_name: str,
        problem,
        algorithm_fn,
        algo_kwargs: dict = None,
        config: dict = None,
        trial: int = 0,
        notes: str = "",
    ) -> dict:
        """Run an algorithm with memory tracking and log the result.

        Args:
            algorithm_fn: callable(problem, **algo_kwargs) -> result dict
            algo_kwargs: keyword args passed to algorithm_fn
        """
        if algo_kwargs is None:
            algo_kwargs = {}

        tracemalloc.start()
        try:
            result = algorithm_fn(problem, **algo_kwargs)
        except Exception as e:
            tracemalloc.stop()
            return {"error": str(e)}

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result['memory_peak_mb'] = peak_mem / (1024 * 1024)

        return self.log_run(dataset_name, problem, result, config=config,
                            trial=trial, notes=notes)

    # ── Load & Query ──────────────────────────────────────────

    def load_all(self) -> list[dict]:
        """Load all logged records."""
        if not os.path.isfile(self.filepath):
            return []
        records = []
        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def to_dataframe(self):
        """Load all records as a pandas DataFrame."""
        import pandas as pd
        records = self.load_all()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        # Flatten config dict into columns prefixed with cfg_
        if 'config' in df.columns:
            cfg_df = pd.json_normalize(df['config']).add_prefix('cfg_')
            df = pd.concat([df.drop('config', axis=1), cfg_df], axis=1)
        return df

    def to_csv(self, csv_path: str = None):
        """Export all records to CSV."""
        df = self.to_dataframe()
        if csv_path is None:
            csv_path = self.filepath.replace('.jsonl', '.csv')
        df.to_csv(csv_path, index=False)
        return csv_path

    # ── Aggregation ───────────────────────────────────────────

    def aggregate(self, group_by=None) -> dict:
        """Aggregate records by (algorithm, dataset).

        Returns dict of:
          {(algorithm, dataset): {
              "count": int,
              "runtime_mean": float, "runtime_std": float,
              "soft_mean": float, "soft_std": float, "soft_min": int, "soft_max": int,
              "hard_mean": float,
              "feasible_rate": float,
              "memory_mean": float,
              ...
          }}
        """
        records = self.load_all()
        if not records:
            return {}

        if group_by is None:
            group_by = ['algorithm', 'dataset']

        groups = defaultdict(list)
        for r in records:
            key = tuple(r.get(k, '') for k in group_by)
            groups[key].append(r)

        import math

        def _stats(values):
            values = [v for v in values if v is not None and not (isinstance(v, float) and math.isinf(v))]
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            n = len(values)
            mean = sum(values) / n
            std = (sum((x - mean) ** 2 for x in values) / max(1, n - 1)) ** 0.5 if n > 1 else 0
            return {"mean": round(mean, 4), "std": round(std, 4),
                    "min": min(values), "max": max(values), "count": n}

        agg = {}
        for key, recs in groups.items():
            a = {"count": len(recs)}

            for metric in ['runtime', 'soft_penalty', 'hard_violations', 'memory_peak_mb',
                           'iterations', 'conflicts', 'room_occupancy', 'period_utilisation',
                           'period_related', 'room_related',
                           'two_in_a_row', 'two_in_a_day', 'period_spread',
                           'non_mixed_durations', 'front_load', 'period_penalty', 'room_penalty']:
                s = _stats([r.get(metric, 0) for r in recs])
                a[f"{metric}_mean"] = s["mean"]
                a[f"{metric}_std"] = s["std"]
                a[f"{metric}_min"] = s["min"]
                a[f"{metric}_max"] = s["max"]

            feasibles = [r.get('feasible', False) for r in recs]
            a["feasible_rate"] = round(sum(feasibles) / len(feasibles), 4) if feasibles else 0

            agg[key] = a

        return agg

    def aggregate_to_dataframe(self, group_by=None):
        """Return aggregated results as a DataFrame."""
        import pandas as pd
        agg = self.aggregate(group_by=group_by)
        if not agg:
            return pd.DataFrame()

        if group_by is None:
            group_by = ['algorithm', 'dataset']

        rows = []
        for key, stats in agg.items():
            row = dict(zip(group_by, key))
            row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_aggregated(self, output_path: str = None, group_by=None):
        """Save aggregated results to JSON."""
        if output_path is None:
            output_path = self.filepath.replace('.jsonl', '_aggregated.json')
        agg = self.aggregate(group_by=group_by)
        # Convert tuple keys to strings for JSON
        serializable = {"|".join(str(k) for k in key): val for key, val in agg.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        return output_path

    def clear(self):
        """Delete all logged records."""
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)