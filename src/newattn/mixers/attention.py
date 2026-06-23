"""Causal multi-head softmax attention (zoology.mixers.attention).

State (the x-axis quantity) is the KV cache: `2 * d_model * seq_len` floats per layer,
which *grows with sequence length* -- attention has effectively unbounded state.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import MixerSpec


class SelfAttention(nn.Module):
    """Causal softmax attention (zoology.mixers.attention.SelfAttention)."""

    def __init__(self, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        seqlen = qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = F.dropout(attention, self.dropout_p if self.training else 0.0)
        out = torch.einsum("bhts,bshd->bthd", attention, v)
        return out


class MHA(nn.Module):
    """Multi-head self-attention (zoology.mixers.attention.MHA)."""

    def __init__(self, d_model: int, num_heads: int = 1, bias: bool = True,
                 dropout: float = 0.0, layer_idx: int | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
        context = self.inner_attn(qkv)
        return self.out_proj(rearrange(context, "... h d -> ... (h d)"))

    def state_size(self, sequence_length: int) -> int:
        # KV cache: keys + values over the sequence
        return 2 * self.d_model * sequence_length


def build(cfg, layer_idx: int) -> nn.Module:
    return MHA(cfg.d_model, num_heads=cfg.num_heads, dropout=cfg.attn_dropout, layer_idx=layer_idx)


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for attention: 4 * n_layers * 2 * d_model * L."""
    return 4 * n_layers * 2 * cfg.d_model * seq_len


def dims_str(cfg) -> str:
    return f"num_heads={cfg.num_heads:>3d}  head_dim={cfg.d_model // cfg.num_heads:>3d}"


SPEC = MixerSpec(
    name="attention",
    build=build,
    state_size_bytes=state_size_bytes,
    dims_str=dims_str,
)
