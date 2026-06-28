"""newattn: a minimal, modular recreation of the Zoology MQAR state-size experiments.

The central experiment from Zoology (Arora, Eyuboglu, et al., "Zoology: Measuring and
Improving Recall in Efficient Language Models") -- state size vs. recall accuracy on the
Multi-Query Associative Recall (MQAR) task -- factored into reusable pieces:

    data       MQAR generator + dataloaders
    model      decoder-only LanguageModel harness (zoology.model.LanguageModel)
    mixers     pluggable sequence mixers (attention / mamba2 / gdn2) behind a registry
    train      the training loop
    sweep      run a state-size sweep + plot the recreated curve
    config     the configuration dataclasses

See `experiments/` for the runnable sweeps and the README for how to run them in Colab.
"""
from __future__ import annotations

from .config import (
    DEFAULT_POINTS,
    ModelConfig,
    MQARTaskConfig,
    SweepConfig,
    SweepPoint,
    TrainParams,
)
from .sweep import run_sweep

__all__ = [
    "DEFAULT_POINTS",
    "ModelConfig",
    "MQARTaskConfig",
    "SweepConfig",
    "SweepPoint",
    "TrainParams",
    "run_sweep",
]
__version__ = "0.1.0"
