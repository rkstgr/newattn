"""Gated DeltaNet 2 sequence mixer -- thin wrapper over fla.layers.GatedDeltaNet2.

Imports flash-linear-attention's layer directly (per the fla README) and exposes the
same (b,l,d) -> (b,l,d) interface and `state_size()` method as the local mixers, so it
drops into the same Block / LanguageModel harness.

The recurrent state is a per-head key x value matrix of shape (head_dim, head_v_dim) for
each of the num_v_heads value heads; summed over heads it has
`num_heads * head_dim * head_v_dim` floats per layer -- independent of sequence length
(bounded state). The fla layer runs its Triton chunk/fused_recurrent kernels (CUDA only).
"""
from __future__ import annotations

import torch.nn as nn

from .base import MixerSpec


class GatedDeltaNet2Mixer(nn.Module):
    """Wrapper over fla.layers.GatedDeltaNet2 with a `state_size()` method.

    fla initializes its own parameters; the LanguageModel skips the generic init for this
    mixer's submodules (see `MixerSpec.self_initializes`).
    """

    def __init__(self, d_model: int, head_dim: int = 64, num_heads: int | None = None,
                 expand_v: float = 1.0, mode: str = "chunk", use_short_conv: bool = True,
                 conv_size: int = 4, allow_neg_eigval: bool = False, layer_idx: int | None = None):
        super().__init__()
        from fla.layers import GatedDeltaNet2  # lazy import: only needed for the gdn2 mixer

        if num_heads is None:
            num_heads = max(1, d_model // head_dim)
        self.d_model = d_model
        self.head_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.num_v_heads = num_heads  # we keep num_v_heads == num_heads
        self.gdn2 = GatedDeltaNet2(
            hidden_size=d_model, head_dim=head_dim, num_heads=num_heads, expand_v=expand_v,
            mode=mode, use_short_conv=use_short_conv, conv_size=conv_size,
            allow_neg_eigval=allow_neg_eigval, layer_idx=layer_idx,
        )

    def forward(self, u):
        # fla layers return (output, attentions, past_key_values); we only need the output.
        out, *_ = self.gdn2(u)
        return out

    def state_size(self, sequence_length: int = 2048) -> int:
        # Per-head delta-rule state S has shape (head_dim, head_v_dim) for each of the
        # num_v_heads value heads -> num_v_heads * head_dim * head_v_dim floats per layer.
        return self.num_v_heads * self.head_dim * self.head_v_dim


def build(cfg, layer_idx: int) -> nn.Module:
    return GatedDeltaNet2Mixer(
        cfg.d_model, head_dim=cfg.gdn2_head_dim, num_heads=cfg.gdn2_num_heads,
        expand_v=cfg.gdn2_expand_v, mode=cfg.gdn2_mode, use_short_conv=cfg.gdn2_use_short_conv,
        conv_size=cfg.gdn2_conv_size, allow_neg_eigval=cfg.gdn2_allow_neg_eigval, layer_idx=layer_idx,
    )


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for Gated DeltaNet 2:
    4 * n_layers * num_heads * head_dim * head_v_dim  (head_v_dim = head_dim * expand_v).

    Independent of sequence length (bounded recurrent state)."""
    num_heads = cfg.gdn2_num_heads or max(1, cfg.d_model // cfg.gdn2_head_dim)
    head_v_dim = int(cfg.gdn2_head_dim * cfg.gdn2_expand_v)
    return 4 * n_layers * num_heads * cfg.gdn2_head_dim * head_v_dim


def dims_str(cfg) -> str:
    nh = cfg.gdn2_num_heads or max(1, cfg.d_model // cfg.gdn2_head_dim)
    hd = cfg.gdn2_head_dim
    hv = int(hd * cfg.gdn2_expand_v)
    return f"num_heads={nh:>3d}  head_dim={hd:>3d}  head_v_dim={hv:>3d}"


SPEC = MixerSpec(
    name="gdn2",
    build=build,
    state_size_bytes=state_size_bytes,
    dims_str=dims_str,
    requires_cuda=True,
    use_amp=True,
    self_initializes=True,
    pip_extra="flash-linear-attention",
)
