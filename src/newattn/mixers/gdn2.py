"""Gated DeltaNet 2 (GDN-2) sequence mixer -- pure PyTorch, no Triton (runs on CPU/any GPU).

A faithful port of `fla.layers.gdn2.GatedDeltaNet2`: identical projections, channel-wise
erase/write gates, per-head log-decay and sigmoid-gated RMSNorm output -- but the matrix-state
recurrence is evaluated by a *vendored pure-PyTorch reference* (`fla.ops.gdn2.naive`) instead of
fla's Triton chunk/fused_recurrent kernels. This is the default `gdn2` mixer so the experiment
runs on a Turing/T4 GPU or on CPU with no Triton dependency; use the `gdn2_triton` mixer for the
fast fla kernels on Ampere+ GPUs.

Per-head matrix state S in R^{K x V} with the GDN-2 update rule
    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
where b in R^K is the channel-wise erase gate, w in R^V the channel-wise write gate, and g in
R^K the channel-wise log-decay.

This mixer runs single-head with `head_dim = d_model` (num_heads = 1), so the state is one full
d_model x (d_model * expand_v) matrix per layer: state size = d_model * d_model * expand_v floats
per layer, independent of sequence length. (Tying head_dim to d_model makes the d_model sweep the
single x-axis knob, as for the other mixers; it scales the state quadratically in d_model. This
differs from `gdn2_triton`, which is multi-head with a fixed head_dim=64.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .base import MixerSpec

# ---------------------------------------------------------------------------------------------
# Pure-PyTorch reference recurrence, vendored verbatim from flash-linear-attention
# (fla/ops/gdn2/naive.py, MIT license, (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li) so the
# mixer carries no Triton/fla import. Update rule on the matrix state S in R^{K x V}:
#   S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
# ---------------------------------------------------------------------------------------------


def naive_recurrent_gdn2(q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False):
    """Token-by-token reference forward pass for GDN-2 (q,k,g,b:[B,T,H,K]; v,w:[B,T,H,V])."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v, g, b, w = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, device=v.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    q = q * scale

    for t in range(T):
        b_q, b_k, b_v = q[:, :, t], k[:, :, t], v[:, :, t]
        b_g, b_b, b_w = g[:, :, t], b[:, :, t], w[:, :, t]
        # Per-channel decay along the K axis of the state.
        h = h * b_g.exp().unsqueeze(-1)
        # Gated read at key (b * k): pull the existing contribution, then write (w * v) minus it.
        erase = ((b_b * b_k).unsqueeze(-1) * h).sum(-2)
        b_v_new = b_w * b_v - erase
        h = h + b_k.unsqueeze(-1) * b_v_new.unsqueeze(-2)
        o[:, :, t] = (b_q.unsqueeze(-1) * h).sum(-2)

    o = o.transpose(1, 2).contiguous().to(v.dtype)
    return o, (h if output_final_state else None)


def naive_chunk_gdn2(q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False,
                     chunk_size=64):
    """Chunkwise PyTorch reference for GDN-2 (WY-representation within-chunk solve)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    BT = chunk_size

    q, k, v, g, b, w = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
    B, H, T, K = k.shape
    V = v.shape[-1]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q, k, v, g, b, w = (F.pad(x, (0, 0, 0, pad_len)) for x in (q, k, v, g, b, w))
    T_pad = q.shape[2]
    NT = T_pad // BT

    q = q * scale

    def chunk(x):
        return x.view(B, H, NT, BT, -1)
    q, k, v, g, b, w = (chunk(x) for x in (q, k, v, g, b, w))

    # Cumulative log-decay within each chunk (per channel).
    g_cum = g.cumsum(-2)
    g_last = g_cum[..., -1:, :]
    k_g = k * g_cum.exp()
    k_g_b = k_g * b

    decay_ij = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3))
    decay_ij_exp = decay_ij.exp()
    tril_mask = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool), diagonal=-1)
    bk = b * k
    T_lower = torch.einsum('bhnik,bhnjk,bhnijk->bhnij', bk, k, decay_ij_exp)
    T_lower = T_lower.masked_fill(~tril_mask, 0.0)
    # Blocked forward substitution to build A_inv = (I + T_lower)^{-1}.
    A_inv = -T_lower
    for i in range(1, BT):
        A_inv[..., i, :i] = A_inv[..., i, :i].clone() + (
            A_inv[..., i, :i, None].clone() * A_inv[..., :i, :i].clone()
        ).sum(-2)
    A_inv = A_inv + torch.eye(BT, device=q.device, dtype=torch.float32)

    u_wy = A_inv @ (w * v)
    w_wy = A_inv @ k_g_b
    k_tail = k * (g_last - g_cum).exp()

    decay_qk = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)).exp()
    causal_mask = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool), diagonal=0)

    S = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()
    o = torch.zeros_like(v)
    for n in range(NT):
        q_n, k_n, g_n = q[:, :, n], k[:, :, n], g_cum[:, :, n]
        g_last_n = g_last[:, :, n].squeeze(-2)
        w_n, u_n, k_tail_n = w_wy[:, :, n], u_wy[:, :, n], k_tail[:, :, n]
        v_new = u_n - w_n @ S
        A_qk = torch.einsum('bhik,bhjk,bhijk->bhij', q_n, k_n, decay_qk[:, :, n]).masked_fill(
            ~causal_mask, 0.0)
        o[:, :, n] = A_qk @ v_new + (q_n * g_n.exp()) @ S
        S = S * g_last_n.unsqueeze(-1).exp() + k_tail_n.transpose(-1, -2) @ v_new

    o = o.reshape(B, H, T_pad, V)[:, :, :T].transpose(1, 2).contiguous().to(v.dtype)
    return o, (S if output_final_state else None)


# ---------------------------------------------------------------------------------------------


class _ShortConv(nn.Module):
    """Depthwise causal short conv + SiLU (fla ShortConvolution, activation='silu')."""

    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):  # x: (b, l, d)
        l = x.shape[1]
        x = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)  # trim right padding -> causal
        return self.act(x)


class GatedDeltaNet2Naive(nn.Module):
    """Pure-PyTorch Gated DeltaNet 2 mixer (mirrors fla.layers.gdn2.GatedDeltaNet2's forward).

    Same (b, l, d) -> (b, l, d) interface and `state_size()` as the other local mixers; the
    matrix-state recurrence is the vendored naive reference above (no Triton).
    """

    def __init__(self, d_model: int, head_dim: int = 64, num_heads: int | None = None,
                 num_v_heads: int | None = None, expand_v: float = 1.0, mode: str = "chunk",
                 use_short_conv: bool = True, conv_size: int = 4, conv_bias: bool = False,
                 allow_neg_eigval: bool = False, norm_eps: float = 1e-5, layer_idx: int | None = None):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, d_model // head_dim)
        if num_v_heads is None:
            num_v_heads = num_heads
        self.d_model = d_model
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_v_heads * self.head_v_dim
        self.mode = mode
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.layer_idx = layer_idx

        # q/k/v projections (+ optional short conv with SiLU; else plain SiLU).
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        if use_short_conv:
            self.q_conv1d = _ShortConv(self.key_dim, conv_size, conv_bias)
            self.k_conv1d = _ShortConv(self.key_dim, conv_size, conv_bias)
            self.v_conv1d = _ShortConv(self.value_dim, conv_size, conv_bias)

        # Decay-gate projection (low-rank bottleneck through head_v_dim) and channel-wise gates.
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(d_model, self.key_dim, bias=False)  # erase gate (K axis)
        self.w_proj = nn.Linear(d_model, self.value_dim, bias=False)  # write gate (V axis)

        # Per-QK-head decay rate (A_log) and per-channel softplus bias (dt_bias), as in fla.
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # Output path: sigmoid-gated RMSNorm (per head, over head_v_dim) + projection.
        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

    def _gated_rmsnorm(self, x, gate):
        # FusedRMSNormGated(activation="sigmoid", norm_before_gate=True): rmsnorm(x)*weight*sigmoid(gate).
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        x = x * self.o_norm_weight * torch.sigmoid(gate.float())
        return x.to(self.o_proj.weight.dtype)

    def forward(self, hidden_states):
        if self.use_short_conv:
            q = self.q_conv1d(self.q_proj(hidden_states))
            k = self.k_conv1d(self.k_proj(hidden_states))
            v = self.v_conv1d(self.v_proj(hidden_states))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Channel-wise log-decay (fp32 for stable cumsums) and channel-wise gates in [0, 1].
        g = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        # Split per head: q, k, g, b on head_k_dim; v, w on head_v_dim.
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        g = -self.A_log.float().exp().unsqueeze(-1) * g  # per-head decay rate on the (H, K) tail

        # Grouped value attention: broadcast QK-side tensors across value-head groups.
        if self.num_v_heads > self.num_heads:
            r = self.num_v_heads // self.num_heads
            q, k, g, b = (repeat(x, "... h d -> ... (h r) d", r=r) for x in (q, k, g, b))
        if self.allow_neg_eigval:
            b = b * 2.0

        # fla calls the kernel with use_qk_l2norm_in_kernel=True; the naive op omits it, so do it here.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        scan = naive_recurrent_gdn2 if self.mode == "fused_recurrent" else naive_chunk_gdn2
        o, _ = scan(q, k, v, g, b, w)

        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self._gated_rmsnorm(o, gate)
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)"))

    def state_size(self, sequence_length: int = 2048) -> int:
        # Per-head delta-rule state S has shape (head_k_dim, head_v_dim) for each value head.
        return self.num_v_heads * self.head_k_dim * self.head_v_dim


def build(cfg, layer_idx: int) -> nn.Module:
    # head_dim (the memory knob) is decoupled from d_model: the q/k/v projections map
    # d_model -> num_heads * head_dim, so the recurrent state can be swept independently of the
    # residual-stream width. num_heads defaults (in __post_init__) to d_model // gdn2_head_dim.
    return GatedDeltaNet2Naive(
        cfg.d_model, head_dim=cfg.gdn2_head_dim, num_heads=cfg.gdn2_num_heads,
        expand_v=cfg.gdn2_expand_v, mode=cfg.gdn2_mode, use_short_conv=cfg.gdn2_use_short_conv,
        conv_size=cfg.gdn2_conv_size, allow_neg_eigval=cfg.gdn2_allow_neg_eigval, layer_idx=layer_idx,
    )


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for the pure-PyTorch Gated DeltaNet 2:
    4 * n_layers * num_v_heads * head_dim * head_v_dim  (head_dim = gdn2_head_dim,
    head_v_dim = head_dim * expand_v, num_v_heads = num_heads).

    Independent of sequence length (bounded recurrent state) and of d_model."""
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
)
