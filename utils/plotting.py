"""utils.plotting — backward-compatibility shim for the old monolith.

Until callers (notebook, main.py, tooling scripts) migrate to
``from utils.plots import ...`` this module re-exports the full historical
API from :mod:`utils.plots`. Delete once the notebook is migrated and there
are no more inbound references.
"""
from __future__ import annotations

from utils.plots import *  # noqa: F401, F403  re-export for legacy callers
from utils.plots import __all__ as _plots_all

__all__ = list(_plots_all)
