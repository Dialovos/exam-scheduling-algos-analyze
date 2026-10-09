"""Smoke-render all README figures in both themes."""
import matplotlib
import pytest

matplotlib.use("Agg")

from utils.plots.readme import FIGURES, THEMES
from utils.plots.shared import load_batch018


@pytest.fixture(scope="module")
def paper_batch():
    return load_batch018()


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", FIGURES)
def test_readme_figure_renders(tmp_path, paper_batch, name, theme):
    path = tmp_path / f"{name}-{theme}.png"
    FIGURES[name](path, theme, batch=paper_batch)
    assert path.is_file()
    assert path.stat().st_size > 0
