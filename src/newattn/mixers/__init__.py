"""Sequence-mixer registry.

Each mixer module (attention, mamba2, gdn2, gdn2_triton, titans) exposes a `SPEC: MixerSpec`. `gdn2`
is the pure-PyTorch Gated DeltaNet 2 (no Triton; CPU/Turing-friendly); `gdn2_triton` is the
fla-kernel version (Ampere+ GPU). To add a new mixer, drop in a module with an `nn.Module`
(forward + `state_size`) and a `SPEC`, then register it here.
"""
from __future__ import annotations

from . import attention, gdn2, gdn2_triton, mamba2, titans
from .base import MixerSpec

MIXERS: dict[str, MixerSpec] = {
    spec.name: spec for spec in (attention.SPEC, mamba2.SPEC, gdn2.SPEC, gdn2_triton.SPEC, titans.SPEC)
}


def get_spec(name: str) -> MixerSpec:
    try:
        return MIXERS[name]
    except KeyError:
        raise ValueError(f"unknown mixer {name!r}; available: {sorted(MIXERS)}") from None


def build_sequence_mixer(cfg, layer_idx: int):
    """Build the sequence mixer selected by `cfg.mixer`."""
    return get_spec(cfg.mixer).build(cfg, layer_idx)


__all__ = ["MixerSpec", "MIXERS", "get_spec", "build_sequence_mixer"]
