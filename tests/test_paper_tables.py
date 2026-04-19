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


def test_t1_csv_has_one_row_per_algo_plus_ip(tmp_path):
    from utils.tables.paper.t1_leaderboard import make_t1
    csv_path, _ = make_t1(tmp_path)
    df = pd.read_csv(csv_path)
    assert len(df) >= 13
    assert "Algorithm" in df.columns
    assert "Mean Rank" in df.columns
    assert "Best On" in df.columns
    assert "IP" in df["Algorithm"].values
    assert "Chain" not in df["Algorithm"].values


def test_t1_cell_format_is_mean_plus_minus_std(tmp_path):
    from utils.tables.paper.t1_leaderboard import make_t1
    csv_path, _ = make_t1(tmp_path)
    df = pd.read_csv(csv_path)
    instance_cols = [c for c in df.columns if c.startswith("exam_comp_set")]
    assert instance_cols, "Should have exam_comp_set1..set8 columns"
    pattern = re.compile(r"^\d+ \+/- \d+\*?$|^-$")
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


def test_t4_writes_csv_and_tex(tmp_path):
    from utils.tables.paper.t4_family_comparison import make_t4
    csv_path, tex_path = make_t4(tmp_path)
    assert csv_path.exists() and tex_path.exists()
    _assert_ascii_only(csv_path)
    _assert_ascii_only(tex_path)


def test_t4_groups_algos_by_family_with_rank_and_wins(tmp_path):
    from utils.tables.paper.t4_family_comparison import make_t4
    csv_path, _ = make_t4(tmp_path)
    df = pd.read_csv(csv_path)
    for col in ("Algorithm", "Family", "Family Rank", "Family Wins"):
        assert col in df.columns
    assert {"Construction", "Trajectory", "Population", "Exact / Hybrid"} <= set(df["Family"])
    instance_cols = [c for c in df.columns if c.startswith("exam_comp_set")]
    pattern = re.compile(r"^\d+ \+/- \d+\*?$|^-$")
    for col in instance_cols:
        for val in df[col].dropna().astype(str):
            assert pattern.match(val), f"T4 cell '{val}' in {col} violates format"
    # Multi-member families must have one starred (best-in-family) cell per
    # instance — proves the marker logic fires. Single-member families
    # (Construction = Greedy alone, Exact/Hybrid = CP-SAT alone) get no
    # marker by design and report "n/a" for rank/wins.
    for fam in df["Family"].unique():
        fam_rows = df[df["Family"] == fam]
        if len(fam_rows) < 2:
            assert (fam_rows["Family Rank"] == "--").all()
            assert (fam_rows["Family Wins"] == "--").all()
            for col in instance_cols:
                cells = fam_rows[col].dropna().astype(str)
                assert not any(c.endswith("*") for c in cells), (
                    f"Solo-member family {fam} should have no '*' marker"
                )
            continue
        for col in instance_cols:
            cells = fam_rows[col].dropna().astype(str)
            if cells.empty or all(c == "-" for c in cells):
                continue
            assert any(c.endswith("*") for c in cells), (
                f"Family {fam} has no '*' marker on {col}"
            )
