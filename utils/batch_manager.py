"""
Isolates each experiment run into its own directory.

Modes:
  - Auto: batch_002_20260405_153022 (incremented ID + timestamp)
  - Manual: batch_003_my_experiment (incremented ID + custom suffix)
  - Browse: list/load any previous batch

Directory layout:
  results/
    batch_018_colab/
    batch_019_<timestamp>/
    batch_020_tuning_sa/
    ...

Each batch dir contains: run_log.jsonl, plots, solution files, etc.

Usage:
    bm = BatchManager()
    bm.new_batch()                # auto: batch_<id>_<timestamp>
    bm.new_batch("tuning_sa")    # manual: batch_<id>_tuning_sa
    bm.list_batches()            # [{id, name, path, created, records}, ...]
    bm.load_batch("batch_018")   # switch to existing batch
    bm.active_dir                # current batch path
    bm.logger                    # ResultsLogger pointed at active batch
"""

import os
import re
import json
from datetime import datetime
from utils.results_logger import ResultsLogger


class BatchManager:
    """Manages isolated result batches under a root results directory."""

    def __init__(self, results_root: str = "results"):
        self.results_root = results_root
        os.makedirs(results_root, exist_ok=True)
        self._active_dir = None
        self._logger = None

    # ── Properties ────────────────────────────────────────────

    @property
    def active_dir(self) -> str | None:
        return self._active_dir

    @property
    def logger(self) -> ResultsLogger | None:
        return self._logger

    # ── Core API ──────────────────────────────────────────────

    def new_batch(self, name: str = None) -> str:
        """Create a new batch directory and set it as active.

        Args:
            name: Optional suffix. If None, uses timestamp (auto mode).
                  e.g. name="tuning_sa" → batch_003_tuning_sa

        Returns:
            Path to the new batch directory.
        """
        next_id = self._next_id()
        if name:
            safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            dirname = f"batch_{next_id:03d}_{safe}"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = f"batch_{next_id:03d}_{ts}"

        path = os.path.join(self.results_root, dirname)
        os.makedirs(path, exist_ok=True)

        # Write metadata
        meta = {"id": next_id, "name": name or "auto", "created": datetime.now().isoformat()}
        with open(os.path.join(path, "batch_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        self._set_active(path)
        print(f"[Batch] Created: {dirname}")
        return path

    def load_batch(self, batch_id_or_name: str | int) -> str:
        """Switch active batch to an existing one.

        Args:
            batch_id_or_name: Batch ID (int), directory name (str),
                              or partial match (e.g. "baseline", "002").

        Returns:
            Path to the loaded batch directory.
        """
        batches = self.list_batches()
        if not batches:
            raise FileNotFoundError("No batches found")

        # Try exact ID match
        if isinstance(batch_id_or_name, int):
            for b in batches:
                if b['id'] == batch_id_or_name:
                    self._set_active(b['path'])
                    print(f"[Batch] Loaded: {b['dirname']}")
                    return b['path']
            raise FileNotFoundError(f"No batch with id={batch_id_or_name}")

        query = str(batch_id_or_name).lower()

        # Try exact dirname match
        for b in batches:
            if b['dirname'].lower() == query:
                self._set_active(b['path'])
                print(f"[Batch] Loaded: {b['dirname']}")
                return b['path']

        # Try partial match (id number or name substring)
        matches = []
        for b in batches:
            if query in b['dirname'].lower():
                matches.append(b)
            elif query.isdigit() and b['id'] == int(query):
                matches.append(b)

        if len(matches) == 1:
            self._set_active(matches[0]['path'])
            print(f"[Batch] Loaded: {matches[0]['dirname']}")
            return matches[0]['path']
        elif len(matches) > 1:
            names = [m['dirname'] for m in matches]
            raise ValueError(f"Ambiguous match for '{batch_id_or_name}': {names}")
        else:
            raise FileNotFoundError(f"No batch matching '{batch_id_or_name}'")

    def list_batches(self) -> list[dict]:
        """List all batch directories with metadata.

        Returns:
            List of dicts: {id, dirname, path, name, created, records}
            sorted by ID ascending.
        """
        batches = []
        for entry in os.listdir(self.results_root):
            full = os.path.join(self.results_root, entry)
            if not os.path.isdir(full):
                continue
            m = re.match(r'^batch_(\d+)', entry)
            if not m:
                continue

            bid = int(m.group(1))
            meta_path = os.path.join(full, "batch_meta.json")
            name = "unknown"
            created = None
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                name = meta.get("name", "unknown")
                created = meta.get("created")

            # Count records
            log_path = os.path.join(full, "run_log.jsonl")
            n_records = 0
            if os.path.isfile(log_path):
                with open(log_path) as f:
                    n_records = sum(1 for line in f if line.strip())

            batches.append({
                "id": bid,
                "dirname": entry,
                "path": full,
                "name": name,
                "created": created,
                "records": n_records,
            })

        batches.sort(key=lambda b: b['id'])
        return batches

    def print_batches(self):
        """Pretty-print all batches."""
        batches = self.list_batches()
        if not batches:
            print("[Batch] No batches found.")
            return

        active = self._active_dir
        print(f"{'':>2} {'ID':>4}  {'Name':<30} {'Records':>8}  {'Created'}")
        print(f"{'':>2} {'─'*4}  {'─'*30} {'─'*8}  {'─'*20}")
        for b in batches:
            marker = " ►" if b['path'] == active else "  "
            created = b['created'][:16].replace('T', ' ') if b['created'] else "—"
            print(f"{marker} {b['id']:>4}  {b['dirname']:<30} {b['records']:>8}  {created}")

    def compare_batches(self, batch_a, batch_b) -> tuple:
        """Load DataFrames from two batches for comparison.

        Args:
            batch_a, batch_b: Batch IDs, names, or partial matches.

        Returns:
            (df_a, df_b) tuple of pandas DataFrames.
        """
        saved_dir = self._active_dir
        saved_logger = self._logger

        def _resolve(query):
            path = self.load_batch(query)
            return self._logger.to_dataframe()

        df_a = _resolve(batch_a)
        df_b = _resolve(batch_b)

        # Restore original active batch
        self._active_dir = saved_dir
        self._logger = saved_logger
        return df_a, df_b

    # ── Convenience ───────────────────────────────────────────

    def ensure_active(self, auto_create: bool = True) -> str:
        """Ensure there's an active batch. Creates one if needed.

        Args:
            auto_create: If True and no active batch, create a new one.

        Returns:
            Path to the active batch directory.
        """
        if self._active_dir and os.path.isdir(self._active_dir):
            return self._active_dir
        if auto_create:
            return self.new_batch()
        raise RuntimeError("No active batch. Call new_batch() or load_batch() first.")

    # ── Internal ──────────────────────────────────────────────

    def _set_active(self, path: str):
        self._active_dir = path
        self._logger = ResultsLogger(os.path.join(path, "run_log.jsonl"))

    def _next_id(self) -> int:
        batches = self.list_batches()
        if not batches:
            return 1
        return max(b['id'] for b in batches) + 1
