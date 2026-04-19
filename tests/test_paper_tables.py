"""Unit + ASCII-only tests for the three paper tables."""
from __future__ import annotations

import re
import pandas as pd
import pytest


ASCII_ONLY = re.compile(r"^[\x00-\x7F]*$")


def _assert_ascii_only(path):
    content = path.read_text()
    assert ASCII_ONLY.match(content), (
        f"Non-ASCII characters in {path}. Table content must be ASCII only."
    )


def test_t1_writes_csv_and_tex(tmp_path):
    from utils.tables.paper.t1_leaderboard import make_t1
    csv_path, tex_path = make_t1(tmp_path)
    assert csv_path.exists() and tex_path.exists()
    _assert_ascii_only(csv_path)
    _assert_ascii_only(tex_path)


def test_t1_csv_has_one_row_per_algo_plus_chain_plus_ip(tmp_path):
    from utils.tables.paper.t1_leaderboard import make_t1
    csv_path, _ = make_t1(tmp_path)
    df = pd.read_csv(csv_path)
    assert len(df) >= 13
    assert "Algorithm" in df.columns
    assert "Mean Rank" in df.columns
    assert "Best On" in df.columns


def test_t1_cell_format_is_mean_plus_minus_std(tmp_path):
    from utils.tables.paper.t1_leaderboard import make_t1
    csv_path, _ = make_t1(tmp_path)
    df = pd.read_csv(csv_path)
    instance_cols = [c for c in df.columns if c.startswith("exam_comp_set")]
    assert instance_cols, "Should have exam_comp_set1..set8 columns"
    pattern = re.compile(r"^\d+ \+/- \d+$|^-$")
    for col in instance_cols:
        for val in df[col].dropna().astype(str):
            assert pattern.match(val), f"Cell '{val}' in {col} violates ASCII mean+/-std format"


def test_t2_writes_csv_and_tex(tmp_path):
    from utils.tables.paper.t2_chain_top5 import make_t2
    csv_path, tex_path = make_t2(tmp_path)
    assert csv_path.exists() and tex_path.exists()
    _assert_ascii_only(csv_path)
    _assert_ascii_only(tex_path)


def test_t2_chain_column_uses_ascii_arrow(tmp_path):
    from utils.tables.paper.t2_chain_top5 import make_t2
    csv_path, _ = make_t2(tmp_path)
    df = pd.read_csv(csv_path)
    assert "Chain" in df.columns
    for chain_str in df["Chain"]:
        assert "\u2192" not in chain_str
        if len(chain_str.split(" -> ")) > 1:
            assert " -> " in chain_str


def test_t2_has_five_rows(tmp_path):
    from utils.tables.paper.t2_chain_top5 import make_t2
    csv_path, _ = make_t2(tmp_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 5


def test_t3_writes_csv_and_tex(tmp_path):
    from utils.tables.paper.t3_partial_adopt import make_t3
    csv_path, tex_path = make_t3(tmp_path)
    assert csv_path.exists() and tex_path.exists()
    _assert_ascii_only(csv_path)
    _assert_ascii_only(tex_path)


def test_t3_has_verdict_column_with_ascii_values(tmp_path):
    from utils.tables.paper.t3_partial_adopt import make_t3
    csv_path, _ = make_t3(tmp_path)
    df = pd.read_csv(csv_path)
    assert "Verdict" in df.columns
    assert set(df["Verdict"].unique()).issubset({"adopted", "reverted"})
