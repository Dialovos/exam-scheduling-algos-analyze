#!/usr/bin/env python3
"""tooling.auto_tuner — backward-compat shim for the old 1421-line monolith.

All code now lives in :mod:`tooling.tuner`; this module re-exports the
historical public API so existing callers (``main.py``, the notebook,
``tooling.optimizers``) keep working. Delete once no inbound references
remain.

Single-dataset mode:
    python -m tooling.auto_tuner instances/exam_comp_set4.exam

Multi-dataset (global) mode:
    python -m tooling.auto_tuner --all-sets
    python -m tooling.auto_tuner --all-sets --synthetic

Resume from checkpoint:
    python -m tooling.auto_tuner --all-sets --resume
"""
from __future__ import annotations

from tooling.tuner import *  # noqa: F401, F403  re-export for legacy callers
from tooling.tuner import __all__ as _tuner_all
from tooling.tuner.cli import main

__all__ = list(_tuner_all)


if __name__ == '__main__':
    main()
