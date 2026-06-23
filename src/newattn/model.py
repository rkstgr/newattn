"""Decoder-only language model harness (mirrors zoology.model.LanguageModel).

token + learnable absolute position embeddings -> `n_layers` pre-norm blocks of
(sequence-mixer + MLP state-mixer) -> final LayerNorm -> weight-tied LM head.
Initialization (`std=0.02`, residual projections scaled by `1/sqrt(2*n_layers)`) matches
`zoology/model.py`. The sequence mixer is chosen by `cfg.mixer` via the mixer registry.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mixers import build_sequence_mixer, get_spec


class MLP(nn.Module):
    """Feed-forward state-mixer (zoology.mixers.mlp.MLP)."""

    def __init__(self, d_model: int, hidden_mult: int = 4, activation=F.gelu):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.activation = activation
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm block: norm -> sequence mixer, then norm -> MLP (state mixer)."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.sequence_mixer = build_sequence_mixer(cfg, layer_idx)
        self.state_mixer = MLP(cfg.d_model, hidden_mult=cfg.mlp_hidden_mult)
        self.dropout1 = nn.Dropout(cfg.embed_dropout if layer_idx == 0 else cfg.resid_dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.dropout2 = nn.Dropout(cfg.resid_dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)

    def forward(self, hidden_states, residual=None):
        dropped = self.dropout1(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm1(residual)
        hidden_states = self.sequence_mixer(hidden_states)

        dropped = self.dropout2(hidden_states)
        residual = dropped + residual
        hidden_states = self.norm2(residual)
        hidden_states = self.state_mixer(hidden_states)
        return hidden_states, residual


class TokenEmbeddings(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if not cfg.learnable_word_embeddings:
            self.word_embeddings.weight.requires_grad = False
        self.max_position_embeddings = cfg.max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.d_model)

    def forward(self, input_ids):
        emb = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            pos = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            emb = emb + self.position_embeddings(pos)
        return emb


def _init_weights(module, n_layers, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        if not getattr(module, "_no_reinit", False):
            nn.init.normal_(module.weight, std=initializer_range)
    # GPT-2-style residual rescaling for projections feeding the residual stream
    for name, p in module.named_parameters():
        if name in ("out_proj.weight", "fc2.weight"):
            nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))


class LanguageModel(nn.Module):
    """Mirrors zoology.model.LanguageModel (sequence_mixer selected by cfg.mixer)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = TokenEmbeddings(cfg)
        self.layers = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.drop_f = nn.Dropout(cfg.resid_dropout)
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        init_fn = partial(_init_weights, n_layers=cfg.n_layers, initializer_range=cfg.initializer_range)
        skip_mixer = get_spec(cfg.mixer).self_initializes  # e.g. fla GatedDeltaNet2 self-initializes
        for name, module in self.named_modules():
            if skip_mixer and "sequence_mixer" in name:
                continue
            init_fn(module)

        if cfg.learnable_word_embeddings:  # weight tying
            self.lm_head.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = dropped + residual
        hidden_states = self.ln_f(residual)
        return self.lm_head(hidden_states)

    def state_size(self, sequence_length: int) -> int:
        # zoology convention: sum the per-layer recurrent/KV state, x4 bytes (float32)
        total = sum(layer.sequence_mixer.state_size(sequence_length) for layer in self.layers)
        return 4 * total
