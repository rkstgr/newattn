"""Mixer registry types.

A `MixerSpec` is the contract every sequence mixer plugs into. To add a new mixer,
implement an `nn.Module` with a `forward(x)->y` and a `state_size(seq_len)` method,
then register a `MixerSpec` for it (see `newattn/mixers/__init__.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class MixerSpec:
    name: str
    build: Callable[..., nn.Module]  # (cfg: ModelConfig, layer_idx: int) -> nn.Module
    state_size_bytes: Callable[..., int]  # (cfg, n_layers, seq_len) -> int   (closed form, bytes)
    dims_str: Callable[..., str]  # (cfg) -> str   (one-line dims for the planning table)
    requires_cuda: bool = False  # mixer only runs on a CUDA GPU
    use_amp: bool = False  # run the forward under bf16 autocast on GPU
    self_initializes: bool = False  # mixer initialises its own params; skip the generic init
    pip_extra: str | None = None  # pip package needed to import the mixer (e.g. flash-linear-attention)
