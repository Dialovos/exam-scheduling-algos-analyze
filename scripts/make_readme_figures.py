"""Render every README figure in both themes from the cached paper data."""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.plots.readme import FIGURES, THEMES  # noqa: E402
from utils.plots.shared import load_batch018  # noqa: E402


def main():
    output = Path("graphs/readme")
    batch = load_batch018()
    for name, render in FIGURES.items():
        for theme in THEMES:
            path = render(output / f"{name}-{theme}.png", theme, batch=batch)
            print(path)


if __name__ == "__main__":
    main()
