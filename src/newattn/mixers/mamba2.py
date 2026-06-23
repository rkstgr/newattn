"""Mamba2 sequence mixer (mirrors zoology.mixers.mamba2.Mamba2 / mamba_ssm Mamba2).

Pure-PyTorch: the SSD scan and the causal depthwise conv are implemented without the
CUDA kernels (mamba_ssm / causal_conv1d / triton) so this runs anywhere, including CPU.

State (the x-axis quantity) is the recurrent SSM hidden state `h` of shape
`(nheads, headdim, d_state)`, i.e. `d_inner * d_state = expand * d_model * d_state` floats
per layer -- *independent of sequence length* (bounded recurrent state).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import MixerSpec


class RMSNormGated(nn.Module):
    """Gated RMSNorm used by Mamba2 (mirrors mamba_ssm RMSNormGated, norm_before_gate=False).

    When a gate `z` is given, normalizes `x * silu(z)` (gate-before-norm). We use a single
    group (ngroups=1), so the norm is taken over the full d_ssm dimension.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def ssd_forward(x, dt, A, B, C, D):
    """Exact (single-chunk, quadratic) Mamba2 state-space-dual scan, in pure PyTorch.

    Evaluates the SSD recurrence
        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t ,    y_t = C_t . h_t + D * x_t
    in closed form as
        y_t = sum_{s<=t} exp(cumsum_t - cumsum_s) * dt_s * (C_t . B_s) * x_s  +  D * x_t ,
    where cumsum is the cumulative sum of dt*A along the sequence. For our seq_len (128)
    the O(L^2) form is exact, simple and fast, and needs no CUDA kernels. It is
    mathematically identical to the chunked `mamba_chunk_scan_combined` used in the
    official implementation.

    Shapes:  x (b,l,h,p)  dt (b,l,h)  A (h,)  B,C (b,l,g,n)  D (h,)  ->  y (b,l,h,p)
    """
    b, l, h, p = x.shape
    g = B.shape[2]
    if g != h:  # broadcast B/C groups across heads (ngroups=1)
        B = B.repeat_interleave(h // g, dim=2)
        C = C.repeat_interleave(h // g, dim=2)

    dA = dt * A  # (b,l,h)  per-step log-decay (<= 0)
    dA_cumsum = torch.cumsum(dA, dim=1)  # (b,l,h)
    cs = rearrange(dA_cumsum, "b l h -> b h l")
    # decay[t,s] = exp(cumsum_t - cumsum_s) for s <= t, else 0 (masked to -inf before exp)
    decay = cs.unsqueeze(-1) - cs.unsqueeze(-2)  # (b,h,t,s)
    causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device))
    decay = decay.masked_fill(~causal, float("-inf")).exp()

    CB = torch.einsum("blhn,bshn->bhls", C, B)  # (b,h,t,s)  C_t . B_s
    M = CB * decay
    xdt = x * dt.unsqueeze(-1)  # fold dt into the input
    y = torch.einsum("bhls,bshp->blhp", M, xdt)  # (b,l,h,p)
    y = y + x * D.view(1, 1, h, 1)  # D skip connection
    return y


def _valid_headdim(d_inner: int, target: int = 64) -> int:
    """Largest head dim <= target that divides d_inner (keeps nheads integral; state size is fixed)."""
    hd = min(target, d_inner)
    while d_inner % hd != 0:
        hd -= 1
    return hd


class Mamba2(nn.Module):
    """Mamba2 sequence mixer.

    Dimensions follow the reference exactly:
        d_inner  = expand * d_model
        nheads   = d_inner // headdim
        conv_dim = d_inner + 2 * ngroups * d_state
        in_proj -> [ z (d_inner), xBC (conv_dim), dt (nheads) ]
    """

    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4, expand: int = 2,
                 headdim: int = 64, ngroups: int = 1, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4, A_init_range=(1.0, 16.0), conv_bias: bool = True,
                 bias: bool = False, layer_idx: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.d_ssm = self.d_inner  # no separate gated-MLP branch (d_ssm == d_inner)
        self.ngroups = ngroups
        self.headdim = _valid_headdim(self.d_inner, headdim)
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        self.layer_idx = layer_idx

        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=self.conv_dim, padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        # dt bias: inverse-softplus of dt ~ exp(U[log dt_min, log dt_max]) (mamba_ssm init)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A (one scalar per head), stored in log space; A = -exp(A_log) < 0 -> stable decay
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = RMSNormGated(self.d_ssm, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, u):
        b, l, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        # causal depthwise conv over the sequence (trim the right padding), then SiLU
        xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        dt = F.softplus(dt + self.dt_bias)  # (b,l,nheads), positive timestep
        A = -torch.exp(self.A_log)  # (nheads,)

        y = ssd_forward(x, dt, A, B, C, self.D)  # (b,l,h,p)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)  # gated RMSNorm
        return self.out_proj(y)

    def state_size(self, sequence_length: int = 2048) -> int:
        # Recurrent SSM state h has shape (nheads, headdim, d_state) -> d_inner * d_state floats.
        # Independent of sequence length (bounded state). Equals zoology's 2 * d_model * d_state
        # for expand=2. (The small conv state, conv_dim * (d_conv-1), is omitted, as in zoology.)
        return self.d_inner * self.d_state


def build(cfg, layer_idx: int) -> nn.Module:
    return Mamba2(cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                  headdim=cfg.headdim, ngroups=cfg.ngroups, layer_idx=layer_idx)


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for Mamba2: 4 * n_layers * (expand*d_model) * d_state.

    Independent of sequence length (bounded recurrent state)."""
    return 4 * n_layers * (cfg.expand * cfg.d_model) * cfg.d_state


def dims_str(cfg) -> str:
    d_inner = cfg.expand * cfg.d_model
    hd = _valid_headdim(d_inner, cfg.headdim)
    return f"d_inner={d_inner:>4d}  headdim={hd:>3d}  nheads={d_inner // hd:>3d}"


SPEC = MixerSpec(
    name="mamba2",
    build=build,
    state_size_bytes=state_size_bytes,
    dims_str=dims_str,
)
