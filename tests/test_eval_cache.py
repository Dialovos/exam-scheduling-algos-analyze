"""Tests for tooling.eval_cache.EvalCache."""
import json
import os
import tempfile
import pytest

from tooling.eval_cache import EvalCache


def test_key_is_deterministic_regardless_of_dict_order():
    c = EvalCache()
    k1 = c.key_for("single", "tabu", "set1.exam", 42, {"a": 1, "b": 2})
    k2 = c.key_for("single", "tabu", "set1.exam", 42, {"b": 2, "a": 1})
    assert k1 == k2


def test_get_returns_none_on_miss():
    c = EvalCache()
    assert c.get("nonexistent_key") is None


def test_put_then_get_roundtrip():
    c = EvalCache()
    key = c.key_for("single", "sa", "set1.exam", 42, {"sa_iters": 10000})
    c.put(key, {"soft_penalty": 1234, "hard_violations": 0})
    hit = c.get(key)
    assert hit == {"soft_penalty": 1234, "hard_violations": 0}


def test_dataset_path_is_normalized_to_basename():
    c = EvalCache()
    k1 = c.key_for("single", "sa", "/abs/path/set1.exam", 42, {"x": 1})
    k2 = c.key_for("single", "sa", "relative/set1.exam", 42, {"x": 1})
    assert k1 == k2


def test_persistence_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        c1 = EvalCache(persist_path=path)
        key = c1.key_for("chain", "hash_abc", "set1.exam", 42, {})
        c1.put(key, {"soft_penalty": 99, "hard_violations": 0})
        c1.save()

        c2 = EvalCache(persist_path=path)
        hit = c2.get(key)
        assert hit is not None
        assert hit["soft_penalty"] == 99


def test_save_is_atomic_on_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        c = EvalCache(persist_path=path)
        c.put(c.key_for("single", "sa", "x.exam", 1, {"a": 1}), {"soft_penalty": 1, "hard_violations": 0})
        c.save()
        with open(path) as f:
            content = f.read()
        assert "soft_penalty" in content


def test_chain_key_includes_full_chain_hash():
    c = EvalCache()
    chain_a = [("sa", {"sa_iters": 1000}), ("gd", {"gd_iters": 2000})]
    chain_b = [("sa", {"sa_iters": 1000}), ("gd", {"gd_iters": 3000})]
    k1 = c.chain_key("set1.exam", 42, chain_a)
    k2 = c.chain_key("set1.exam", 42, chain_b)
    assert k1 != k2


def test_hit_counter_tracks_usage():
    c = EvalCache()
    key = c.key_for("single", "sa", "x.exam", 1, {"a": 1})
    c.put(key, {"soft_penalty": 1, "hard_violations": 0})
    assert c.stats() == {"hits": 0, "misses": 0, "size": 1}
    c.get(key)
    c.get("miss_key")
    assert c.stats() == {"hits": 1, "misses": 1, "size": 1}
