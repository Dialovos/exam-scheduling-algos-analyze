"""Persistent hash-keyed cache for evaluation results.

Used by auto_tuner to avoid re-running identical (algo, dataset, seed, params)
configurations. The cache is process-local (main process) and checked before
submitting work to ProcessPoolExecutor — worker processes do not see it.

Key design:
    single-algo key: ('single', algo, basename(dataset), seed, params_hash)
    chain key:       ('chain', basename(dataset), seed, chain_hash)

params_hash is sha1 of the sorted param items as JSON.
chain_hash is sha1 of the sorted chain steps as JSON.

Keys are strings (JSON-friendly for on-disk persistence).
"""
import hashlib
import json
import os
from pathlib import Path


def _hash_params(params: dict) -> str:
    """Return a stable sha1 hex digest of a params dict."""
    items = sorted((str(k), v) for k, v in (params or {}).items())
    blob = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _hash_chain(chain_steps) -> str:
    """Return a stable sha1 hex digest of a chain [(algo, params), ...]."""
    norm = [(a, sorted((str(k), v) for k, v in (p or {}).items())) for a, p in chain_steps]
    blob = json.dumps(norm, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class EvalCache:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path
        self._store: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        if persist_path and os.path.isfile(persist_path):
            try:
                with open(persist_path) as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}

    @staticmethod
    def key_for(kind: str, algo: str, dataset: str, seed: int, params: dict) -> str:
        ds = Path(dataset).name  # basename — abs vs relative paths collapse
        return f"{kind}|{algo}|{ds}|{seed}|{_hash_params(params)}"

    @staticmethod
    def chain_key(dataset: str, seed: int, chain_steps) -> str:
        ds = Path(dataset).name
        return f"chain|{ds}|{seed}|{_hash_chain(chain_steps)}"

    def get(self, key: str) -> dict | None:
        hit = self._store.get(key)
        if hit is None:
            self._misses += 1
            return None
        self._hits += 1
        return hit

    def put(self, key: str, result: dict) -> None:
        if result is None:
            return
        self._store[key] = result

    def save(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        tmp = self.persist_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._store, f, default=str)
        os.replace(tmp, self.persist_path)

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}
