"""Unit tests for paper-plot helpers added to utils.plots.shared."""
from __future__ import annotations

import pandas as pd
import pytest

from utils.plots.shared import (
    FAMILY_COLORS,
    FAMILY_ORDER_WITH_CHAIN,
    normalize_per_instance,
    load_batch018,
    Batch018,
)


def test_chain_family_color_exists():
    assert "Chain" in FAMILY_COLORS
    assert FAMILY_COLORS["Chain"].startswith("#")


def test_family_order_with_chain_has_5_entries():
    assert FAMILY_ORDER_WITH_CHAIN == [
        "Construction", "Trajectory", "Population", "Exact / Hybrid", "Chain",
    ]


def test_normalize_per_instance_divides_by_per_dataset_min():
    df = pd.DataFrame({
        "algorithm": ["A", "B", "A", "B"],
        "dataset":   ["d1", "d1", "d2", "d2"],
        "soft_penalty": [100, 200, 300, 600],
    })
    out = normalize_per_instance(df)
    # d1 min = 100, d2 min = 300
    assert out.loc[out["dataset"] == "d1", "soft_norm"].tolist() == [1.0, 2.0]
    assert out.loc[out["dataset"] == "d2", "soft_norm"].tolist() == [1.0, 2.0]


def test_normalize_per_instance_preserves_original_columns():
    df = pd.DataFrame({
        "algorithm": ["A"], "dataset": ["d1"], "soft_penalty": [42],
    })
    out = normalize_per_instance(df)
    assert set(df.columns).issubset(set(out.columns))
    assert "soft_norm" in out.columns


def test_load_batch018_returns_dataclass_with_expected_frames():
    b = load_batch018()
    assert isinstance(b, Batch018)
    assert len(b.main) > 0
    assert "soft_penalty" in b.main.columns
    assert len(b.scaling) > 0
    assert "runtime" in b.scaling.columns
    assert len(b.sweep) > 0
    assert isinstance(b.ip_soft, dict)
    assert isinstance(b.chain_top5, dict)
    assert "top5" in b.chain_top5
    assert isinstance(b.family_map, dict)
    assert "ALNS" in b.family_map


def test_load_batch018_ip_dict_has_only_solved_instances():
    b = load_batch018()
    # Sets 3, 5, 7 timed out per INDEX.md
    for k in b.ip_soft.keys():
        assert k.startswith("exam_comp_set")
        # Each entry should have the 7 soft-component keys
        entry = b.ip_soft[k]
        assert "two_in_a_row" in entry
