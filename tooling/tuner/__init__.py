"""tooling.tuner — the auto-tuner carved out of the old 1421-line monolith.

Package layout:
    * :mod:`tooling.tuner.search_spaces` — knob ranges + default params + eval seeds
    * :mod:`tooling.tuner.binary`        — locate/build the C++ solver binary
    * :mod:`tooling.tuner.eval`          — trial execution + multi-seed/ds scoring
    * :mod:`tooling.tuner.sampling`      — random param sampling + chain mutation
    * :mod:`tooling.tuner.checkpoint`    — atomic JSON checkpoint for resumes
    * :mod:`tooling.tuner.synthetic`     — synthetic dataset generation
    * :mod:`tooling.tuner.core`          — the :class:`AutoTuner` orchestrator
    * :mod:`tooling.tuner.cli`           — argparse wrapper + ``main()``

Importing from :mod:`tooling.auto_tuner` still works — that module is a
thin shim that re-exports everything from here for historical callers.
"""
from __future__ import annotations

from tooling.tuner.binary import find_or_build_binary
from tooling.tuner.checkpoint import Checkpoint
from tooling.tuner.cli import main
from tooling.tuner.core import AutoTuner
from tooling.tuner.eval import (
    compute_score,
    eval_chain_multi_seed,
    eval_chain_multi_seed_datasets,
    eval_chain_on_datasets,
    eval_multi_seed,
    eval_multi_seed_datasets,
    eval_on_datasets,
    run_chain,
    run_single_algo,
)
from tooling.tuner.sampling import (
    _sample_val,
    mutate_chain,
    perturb,
    random_chain,
    sample_random,
)
from tooling.tuner.search_spaces import (
    DEFAULT_PARAMS,
    EVAL_SEEDS,
    FALLBACK_PARAMS,
    SEARCH_SPACES,
    TUNABLE_ALGOS,
)
from tooling.tuner.synthetic import generate_synthetic_dataset

__all__ = [
    # Classes
    "AutoTuner", "Checkpoint",
    # CLI
    "main",
    # Search space metadata
    "SEARCH_SPACES", "DEFAULT_PARAMS", "FALLBACK_PARAMS",
    "TUNABLE_ALGOS", "EVAL_SEEDS",
    # Binary
    "find_or_build_binary",
    # Evaluation helpers
    "compute_score", "run_single_algo", "run_chain",
    "eval_on_datasets", "eval_chain_on_datasets",
    "eval_multi_seed", "eval_multi_seed_datasets",
    "eval_chain_multi_seed", "eval_chain_multi_seed_datasets",
    # Sampling / mutation
    "sample_random", "perturb", "random_chain", "mutate_chain", "_sample_val",
    # Synthetic data
    "generate_synthetic_dataset",
]
