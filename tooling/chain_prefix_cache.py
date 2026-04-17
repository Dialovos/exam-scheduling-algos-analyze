"""On-disk prefix `.sln` cache for chain warm-starts.

When chain A is ``[SA, GD, LAHC]`` and chain B is ``[SA, GD, KEMPE]``, both
share prefix ``[SA, GD]``. Running A first, we persist the post-``[SA, GD]``
.sln; running B, we skip those two steps and warm-start directly from the
cached solution.

Layout::

    cache_dir/
        <prefix_hash>__<ds_basename>__<seed>/
            final.sln
            result.json    # {soft_penalty, hard_violations, runtime, per_step, ...}

LRU eviction by entry mtime. Process-local in-memory state is limited to
the cache dir path and the configured caps — all lookups hit the filesystem
so multiple tuner processes can safely share a cache_dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def _hash_prefix(prefix_steps) -> str:
    norm = [(a, sorted((str(k), v) for k, v in (p or {}).items()))
            for a, p in prefix_steps]
    blob = json.dumps(norm, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class PrefixCache:
    def __init__(self, cache_dir: str, max_entries: int = 500,
                 max_bytes: int = 2 * 1024 * 1024 * 1024):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.enabled = True
        # Dir is created lazily by :meth:`store` so construction doesn't
        # conflict with the tuner's stale-dir cleanup at __init__ time.

    @staticmethod
    def check_disk(out_dir: str, min_free_gb: float = 5.0) -> bool:
        """Return True if the filesystem hosting *out_dir* has >= *min_free_gb*."""
        try:
            free = shutil.disk_usage(out_dir).free
            return free >= min_free_gb * 1024 ** 3
        except OSError:
            return False

    def _entry_dir(self, prefix_steps, dataset: str, seed: int) -> str:
        ph = _hash_prefix(prefix_steps)
        ds = Path(dataset).name
        return os.path.join(self.cache_dir, f'{ph}__{ds}__{seed}')

    def lookup(self, prefix_steps, dataset: str, seed: int) -> dict | None:
        """Return cached result dict with ``sln_path`` key, or None on miss."""
        if not self.enabled or not prefix_steps:
            return None
        d = self._entry_dir(prefix_steps, dataset, seed)
        result_path = os.path.join(d, 'result.json')
        sln_path = os.path.join(d, 'final.sln')
        if not (os.path.isfile(result_path) and os.path.isfile(sln_path)):
            return None
        try:
            with open(result_path) as f:
                res = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        # Refresh mtime for LRU recency
        try:
            os.utime(d, None)
        except OSError:
            pass
        res['sln_path'] = sln_path
        return res

    def store(self, prefix_steps, dataset: str, seed: int,
              sln_path: str, result: dict) -> None:
        """Persist a prefix's .sln and result JSON, evicting old entries as needed."""
        if not self.enabled or not prefix_steps:
            return
        if not sln_path or not os.path.isfile(sln_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        d = self._entry_dir(prefix_steps, dataset, seed)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, 'final.sln')
        try:
            shutil.copyfile(sln_path, dest)
            with open(os.path.join(d, 'result.json'), 'w') as f:
                json.dump({k: v for k, v in result.items() if k != 'sln_path'},
                          f, default=str)
        except OSError:
            return
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        try:
            entries = [os.path.join(self.cache_dir, e)
                       for e in os.listdir(self.cache_dir)
                       if os.path.isdir(os.path.join(self.cache_dir, e))]
        except OSError:
            return
        entries.sort(key=lambda p: os.path.getmtime(p))
        while len(entries) > self.max_entries:
            shutil.rmtree(entries[0], ignore_errors=True)
            entries.pop(0)

        def _dir_size(p: str) -> int:
            total = 0
            for root, _, files in os.walk(p):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return total

        total_bytes = sum(_dir_size(e) for e in entries)
        while total_bytes > self.max_bytes and entries:
            victim = entries.pop(0)
            total_bytes -= _dir_size(victim)
            shutil.rmtree(victim, ignore_errors=True)
