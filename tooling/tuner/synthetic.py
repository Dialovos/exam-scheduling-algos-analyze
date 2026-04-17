"""Synthetic dataset generation for overfitting-robust tuning.

Mixing a synthetic instance into the eval subset helps prevent the tuner
from memorising ITC 2007 quirks — a guard against overfitting when the
paper's "hidden" instances would otherwise be the only OOD signal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def generate_synthetic_dataset(output_dir, num_exams=500, preset='competition',
                               seed=42):
    """Generate a synthetic dataset and return its path.

    Cached: if the target file already exists we return it directly, so
    repeated tuner invocations don't re-pay the generation cost.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.generator import generate_synthetic, write_itc2007_format

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'synthetic_{preset}_{num_exams}.exam')
    if os.path.isfile(path):
        return path

    print(f"[AutoTuner] Generating synthetic ({num_exams} exams, {preset})...")
    prob = generate_synthetic(num_exams=num_exams, preset=preset, seed=seed)
    write_itc2007_format(prob, path)
    print(f"[AutoTuner] Synthetic: {path}")
    return path
