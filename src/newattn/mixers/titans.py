"""Titans neural-memory sequence mixer -- pure PyTorch, no Triton (runs on CPU/any GPU).

A PyTorch port of the JAX/Equinox reference `linattn.models.titans.Titans`. Unlike the
linear-attention mixers (mamba2/gdn2) whose per-head state is a *matrix* updated by a linear
recurrence, Titans' per-head fast memory is a small two-layer MLP whose *weights* are the
recurrent state. At every token the weights take one inner-loop gradient-descent step (with
momentum and weight decay) on a memorisation loss `||MLP_k(k_t) - v_t||^2`, then the token is
read out as `MLP(q_t)`:

    pre      = k_t @ W1 ;  h = silu(pre) ;  pred = h @ W2
    residual = v_t - pred                                   # memorisation error
    gW2      = -h (x) residual                              # grad wrt the memorisation loss
    gW1      = -k_t (x) ((W2 @ residual) * silu'(pre))
    mW{1,2}  = nu * mW{1,2} + gW{1,2}                       # momentum (data-dependent decay nu)
    W{1,2}   = alpha * W{1,2} - beta * mW{1,2}              # weight decay alpha + inner LR beta
    o_t      = silu(q_t @ W1) @ W2                          # read with the updated memory

Per head the memory is W1 in R^{head_dim x mem_hidden} and W2 in R^{mem_hidden x head_dim}
(mem_hidden = memory_mult * head_dim), so the recurrent state is 2 * head_dim * mem_hidden floats
per head, *independent of sequence length*. The MLP non-linearity means there is no chunked
closed form, so the scan is an explicit token-by-token loop (as for the gdn2 `fused_recurrent`
mode); for the MQAR seq_len (128) this is fast and exact.

Like the pure-PyTorch `gdn2` mixer this runs single-head with `head_dim = d_model`
(`titans_num_heads = 1`) by default, so the state grows quadratically in d_model and d_model is
the single state-size x-axis knob; set `titans_num_heads > 1` for a multi-head memory with
`head_dim = d_model // num_heads`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import MixerSpec


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    """Derivative of SiLU(x) = x * sigmoid(x)."""
    s = torch.sigmoid(x)
    return s + x * s * (1.0 - s)


class _ShortConv(nn.Module):
    """Depthwise causal short conv + SiLU (matches the reference's causal_dwconv -> silu)."""

    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):  # x: (b, l, d)
        l = x.shape[1]
        x = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)  # trim right padding -> causal
        return self.act(x)


class Titans(nn.Module):
    """Titans mixer with a per-head two-layer MLP as test-time-trained fast memory.

    Same `(b, l, d) -> (b, l, d)` interface and `state_size()` as the other local mixers. The
    q/k/v/o projections are plain `nn.Linear` (initialised by the generic harness init); the
    gate parameters (Wbeta/Wnu/Walpha + biases, dt_logit) and the memory MLP init (mem_W1/mem_W2)
    are `nn.Parameter`s carrying the reference's custom init (left untouched by the harness init,
    as with mamba2's `A_log`/`dt_bias`).
    """

    def __init__(self, d_model: int, num_heads: int = 1, memory_mult: int = 4,
                 max_inner_lr: float = 0.05, conv_size: int = 4, use_short_conv: bool = True,
                 layer_idx: int | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.memory_hidden = memory_mult * self.head_dim
        self.max_inner_lr = max_inner_lr
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        # q/k/v projections (+ optional short conv with SiLU; else plain SiLU), and output proj.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if use_short_conv:
            self.q_conv1d = _ShortConv(d_model, conv_size)
            self.k_conv1d = _ShortConv(d_model, conv_size)
            self.v_conv1d = _ShortConv(d_model, conv_size)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head inner-loop gates (computed from the raw input, as in the reference):
        #   beta  = max_inner_lr * sigmoid(x @ Wbeta)          -- inner learning rate
        #   nu    = sigmoid(x @ Wnu + bnu)                     -- momentum retention (bnu init = 1)
        #   alpha = exp(-softplus(dt_logit) * sigmoid(x @ Walpha + balpha))  -- weight decay
        s = 1.0 / (d_model ** 0.5)
        self.Wbeta = nn.Parameter(torch.randn(d_model, num_heads) * s)
        self.Wnu = nn.Parameter(torch.randn(d_model, num_heads) * s)
        self.bnu = nn.Parameter(torch.ones(num_heads))
        self.Walpha = nn.Parameter(torch.randn(d_model, num_heads) * s)
        self.balpha = nn.Parameter(torch.zeros(num_heads))
        self.dt_logit = nn.Parameter(torch.full((num_heads,), -10.0))

        # Memory MLP initial weights (the recurrent state is reset to these per sequence).
        s1 = 1.0 / (self.head_dim ** 0.5)
        s2 = 1.0 / (self.memory_hidden ** 0.5)
        self.mem_W1 = nn.Parameter(torch.randn(num_heads, self.head_dim, self.memory_hidden) * s1)
        self.mem_W2 = nn.Parameter(torch.randn(num_heads, self.memory_hidden, self.head_dim) * s2)

    def _project(self, x):
        H = self.num_heads
        if self.use_short_conv:
            q = self.q_conv1d(self.q_proj(x))
            k = self.k_conv1d(self.k_proj(x))
            v = self.v_conv1d(self.v_proj(x))
        else:
            q = F.silu(self.q_proj(x))
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=H) for t in (q, k, v))
        # L2-normalise q/k (the reference divides by norm + 1e-6).
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        beta = self.max_inner_lr * torch.sigmoid(x @ self.Wbeta)  # (b, l, h)
        nu = torch.sigmoid(x @ self.Wnu + self.bnu)
        g = torch.sigmoid(x @ self.Walpha + self.balpha)
        dt = F.softplus(self.dt_logit)
        alpha = torch.exp(-dt * g)
        return q, k, v, beta, nu, alpha

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        q, k, v, beta, nu, alpha = self._project(hidden_states)

        # Run the inner-loop recurrence in fp32 for stable test-time gradient steps.
        out_dtype = hidden_states.dtype
        q, k, v = q.float(), k.float(), v.float()
        beta, nu, alpha = beta.float(), nu.float(), alpha.float()  # (b, l, h)

        # Per-(batch, head) memory state, reset to the learned init each sequence.
        W1 = self.mem_W1.float().expand(B, -1, -1, -1).clone()  # (b, h, head_dim, mem_hidden)
        W2 = self.mem_W2.float().expand(B, -1, -1, -1).clone()  # (b, h, mem_hidden, head_dim)
        mW1 = torch.zeros_like(W1)
        mW2 = torch.zeros_like(W2)

        outs = []
        for t in range(T):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]  # (b, h, head_dim)
            beta_t = beta[:, t, :, None, None]                  # (b, h, 1, 1)
            nu_t = nu[:, t, :, None, None]
            alpha_t = alpha[:, t, :, None, None]

            pre = torch.einsum("bhd,bhde->bhe", k_t, W1)        # (b, h, mem_hidden)
            h = F.silu(pre)
            pred = torch.einsum("bhe,bhed->bhd", h, W2)         # (b, h, head_dim)
            residual = v_t - pred

            grad_W2 = -torch.einsum("bhe,bhd->bhed", h, residual)
            hidden_err = torch.einsum("bhed,bhd->bhe", W2, residual) * silu_grad(pre)
            grad_W1 = -torch.einsum("bhd,bhe->bhde", k_t, hidden_err)

            mW1 = nu_t * mW1 + grad_W1
            mW2 = nu_t * mW2 + grad_W2
            W1 = alpha_t * W1 - beta_t * mW1
            W2 = alpha_t * W2 - beta_t * mW2

            pre_q = torch.einsum("bhd,bhde->bhe", q_t, W1)
            o_t = torch.einsum("bhe,bhed->bhd", F.silu(pre_q), W2)  # (b, h, head_dim)
            outs.append(o_t)

        out = torch.stack(outs, dim=2)  # (b, h, l, head_dim)
        out = rearrange(out, "b h l d -> b l (h d)").to(out_dtype)
        return self.o_proj(out)

    def state_size(self, sequence_length: int = 2048) -> int:
        # Fast-weight memory per head: W1 (head_dim, mem_hidden) + W2 (mem_hidden, head_dim).
        # Independent of sequence length (the momentum buffers mW1/mW2 are omitted, as the small
        # conv state is omitted for the other mixers).
        return self.num_heads * 2 * self.head_dim * self.memory_hidden


def build(cfg, layer_idx: int) -> nn.Module:
    return Titans(
        cfg.d_model, num_heads=cfg.titans_num_heads, memory_mult=cfg.titans_memory_mult,
        max_inner_lr=cfg.titans_max_inner_lr, conv_size=cfg.titans_conv_size,
        use_short_conv=cfg.titans_use_short_conv, layer_idx=layer_idx,
    )


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for Titans:
    4 * n_layers * num_heads * 2 * head_dim * mem_hidden  (head_dim = d_model // num_heads,
    mem_hidden = memory_mult * head_dim). With the default single head this is
    8 * memory_mult * n_layers * d_model**2.

    Independent of sequence length (bounded recurrent fast-weight state)."""
    head_dim = cfg.d_model // cfg.titans_num_heads
    mem_hidden = cfg.titans_memory_mult * head_dim
    return 4 * n_layers * cfg.titans_num_heads * 2 * head_dim * mem_hidden


def dims_str(cfg) -> str:
    head_dim = cfg.d_model // cfg.titans_num_heads
    mem_hidden = cfg.titans_memory_mult * head_dim
    return f"num_heads={cfg.titans_num_heads:>3d}  head_dim={head_dim:>3d}  mem_hidden={mem_hidden:>4d}"


SPEC = MixerSpec(
    name="titans",
    build=build,
    state_size_bytes=state_size_bytes,
    dims_str=dims_str,
)
