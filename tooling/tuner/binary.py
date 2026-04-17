"""Locate (or build) the C++ solver binary exactly once per tuner run."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def find_or_build_binary():
    """Return a path to ``cpp/build/exam_solver``, compiling it if needed.

    Raises ``RuntimeError`` if source is missing or compilation fails —
    the tuner is useless without a solver, so we fail loudly rather than
    limping along.
    """
    root = Path(__file__).resolve().parent.parent.parent
    binary = root / 'cpp' / 'build' / 'exam_solver'
    if binary.is_file() and os.access(binary, os.X_OK):
        return str(binary)
    src = root / 'cpp' / 'src' / 'main.cpp'
    if not src.is_file():
        raise RuntimeError(f"Cannot find {src}")
    print("[AutoTuner] Compiling C++ solver...")
    binary.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ['g++', '-O3', '-std=c++20', '-o', str(binary), str(src)],
        capture_output=True, text=True, timeout=120,
        cwd=str(root),
    )
    if r.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{r.stderr}")
    print("[AutoTuner] Compiled successfully")
    return str(binary)
