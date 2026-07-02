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

`head_dim` (the memory knob) is decoupled from d_model: the q/k/v projections map
d_model -> num_heads * head_dim, so the recurrent state (~ memory_mult * head_dim^2 per head)
can be swept independently of the residual-stream width. With `titans_head_dim = None` it falls
back to `head_dim = d_model // num_heads` (the original single-/multi-head behavior).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

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
                 head_dim: int | None = None, max_inner_lr: float = 0.05, conv_size: int = 4,
                 use_short_conv: bool = True, mode: str = "recurrent", chunk_size: int = 64,
                 update_norm: str = "none", weight_norm: bool = False, update_eps: float = 1e-3,
                 checkpoint_chunks: bool = True, layer_idx: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # head_dim decoupled from d_model; default keeps the original d_model // num_heads.
        if head_dim is None:
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads when head_dim is None"
            head_dim = d_model // num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim  # total q/k/v width
        self.memory_hidden = memory_mult * self.head_dim
        self.max_inner_lr = max_inner_lr
        self.use_short_conv = use_short_conv
        self.mode = mode
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        # Inner-loop stabilization (see config). "none"+False reproduces the exp004 baseline exactly.
        assert update_norm in ("none", "frobenius"), f"unknown titans update_norm {update_norm!r}"
        self.update_norm = update_norm
        self.weight_norm = weight_norm
        self.update_eps = update_eps
        self.stabilized = update_norm != "none" or weight_norm
        self.layer_idx = layer_idx

        # q/k/v projections (+ optional short conv with SiLU; else plain SiLU), and output proj.
        # Projections map d_model <-> inner_dim = num_heads * head_dim (decoupled from d_model).
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        if use_short_conv:
            self.q_conv1d = _ShortConv(self.inner_dim, conv_size)
            self.k_conv1d = _ShortConv(self.inner_dim, conv_size)
            self.v_conv1d = _ShortConv(self.inner_dim, conv_size)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

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
        # L2-normalise q/k. The baseline divides by (norm + 1e-6), whose backward is 0/0 (NaN) when a
        # head vector is exactly zero; the stabilized path puts eps *inside* the sqrt (finite backward).
        if self.stabilized:
            q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + self.update_eps ** 2)
            k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + self.update_eps ** 2)
        else:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        beta = self.max_inner_lr * torch.sigmoid(x @ self.Wbeta)  # (b, l, h)
        nu = torch.sigmoid(x @ self.Wnu + self.bnu)
        g = torch.sigmoid(x @ self.Walpha + self.balpha)
        dt = F.softplus(self.dt_logit)
        alpha = torch.exp(-dt * g)
        return q, k, v, beta, nu, alpha

    def forward(self, hidden_states):
        q, k, v, beta, nu, alpha = self._project(hidden_states)

        # Run the inner-loop recurrence in fp32 for stable test-time gradient steps.
        out_dtype = hidden_states.dtype
        q, k, v = q.float(), k.float(), v.float()
        beta, nu, alpha = beta.float(), nu.float(), alpha.float()  # (b, l, h)

        if self.mode == "chunk":
            out = self._chunk_scan(q, k, v, beta, nu, alpha)  # (b, h, l, head_dim)
        else:
            out = self._recurrent_scan(q, k, v, beta, nu, alpha)

        out = rearrange(out, "b h l d -> b l (h d)").to(out_dtype)
        return self.o_proj(out)

    def _norm_grad(self, g):
        """Frobenius-normalize an inner-loop gradient matrix to a unit-norm update *direction*.

        Decouples the step magnitude from ||grad|| (hence from ||h||^2 ~ mem_hidden), so the
        effective inner LR is curvature/width-invariant (Hyperball arXiv:2603.28743; LaCT uses
        Muon for the same end). The eps floor damps the step as the memorisation residual -> 0.
        Normalizing over the trailing matrix dims keeps it per (batch, head[, token]), so it is
        linear-compatible with the chunked closed form and identical to the per-token loop at C=1.
        """
        if self.update_norm == "none":
            return g
        return g / (g.norm(dim=(-2, -1), keepdim=True) + self.update_eps)

    def _ball(self, W, home_norm):
        """Project a fast-weight matrix onto the Frobenius ball of radius = its home (init) norm.

        A no-op while inside the ball, so normal memorisation dynamics are untouched; it only caps
        runaway growth (the "magnitude explosion" of accumulating fast-weight updates noted by LaCT
        arXiv:2505.23884). `home_norm` is pre-shaped to broadcast over W's trailing matrix dims.
        """
        if not self.weight_norm:
            return W
        scale = (home_norm / (W.norm(dim=(-2, -1), keepdim=True) + 1e-12)).clamp(max=1.0)
        return W * scale

    def _home_norms(self, extra_dims: int):
        """Per-head Frobenius norms of the memory init weights, shaped to broadcast over W matrices.

        extra_dims = number of axes between the head axis and the trailing (d, m) matrix axes
        (0 for the recurrent (b, h, d, m); 1 for the chunked (b, h, c, d, m))."""
        shape = (1, self.num_heads) + (1,) * extra_dims + (1, 1)
        w1 = self.mem_W1.detach().norm(dim=(-2, -1)).view(shape)
        w2 = self.mem_W2.detach().norm(dim=(-2, -1)).view(shape)
        return w1, w2

    def _recurrent_scan(self, q, k, v, beta, nu, alpha):
        """Exact token-by-token inner-loop scan. q/k/v: (b, h, l, head_dim); gates: (b, l, h)."""
        B, _, T, _ = q.shape
        # Per-(batch, head) memory state, reset to the learned init each sequence.
        W1 = self.mem_W1.float().expand(B, -1, -1, -1).clone()  # (b, h, head_dim, mem_hidden)
        W2 = self.mem_W2.float().expand(B, -1, -1, -1).clone()  # (b, h, mem_hidden, head_dim)
        mW1 = torch.zeros_like(W1)
        mW2 = torch.zeros_like(W2)
        w1_home, w2_home = self._home_norms(extra_dims=0)  # (1, h, 1, 1) each; only used if weight_norm

        out = q.new_empty(B, self.num_heads, T, self.head_dim)
        for t in range(T):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]  # (b, h, head_dim)
            beta_t = beta[:, t, :, None, None]                  # (b, h, 1, 1)
            nu_t = nu[:, t, :, None, None]
            alpha_t = alpha[:, t, :, None, None]

            pre = torch.einsum("bhd,bhde->bhe", k_t, W1)        # (b, h, mem_hidden)
            h = F.silu(pre)
            pred = torch.einsum("bhe,bhed->bhd", h, W2)         # (b, h, head_dim)
            residual = v_t - pred

            grad_W2 = self._norm_grad(-torch.einsum("bhe,bhd->bhed", h, residual))
            hidden_err = torch.einsum("bhed,bhd->bhe", W2, residual) * silu_grad(pre)
            grad_W1 = self._norm_grad(-torch.einsum("bhd,bhe->bhde", k_t, hidden_err))

            mW1 = nu_t * mW1 + grad_W1
            mW2 = nu_t * mW2 + grad_W2
            W1 = self._ball(alpha_t * W1 - beta_t * mW1, w1_home)
            W2 = self._ball(alpha_t * W2 - beta_t * mW2, w2_home)

            pre_q = torch.einsum("bhd,bhde->bhe", q_t, W1)
            out[:, :, t] = torch.einsum("bhe,bhed->bhd", F.silu(pre_q), W2)  # (b, h, head_dim)
        return out

    def _chunk_step(self, q_n, k_n, v_n, beta_n, An, Bn, coefM, coefW, W1, W2, mW1, mW2,
                    w1_home, w2_home):
        """One mini-batch chunk of the scan. Returns (out_chunk, W1, W2, mW1, mW2) where the
        states are the carries from the chunk's last token. Run under gradient checkpointing
        (see `_chunk_scan`): the (b, h, c, d, m) per-token weight tensors created here dominate
        training memory, so they are recomputed in backward instead of stored."""
        # Inner-loop gradients of ||MLP_k(k) - v||^2 at the chunk-start weights (all tokens at once).
        pre = torch.einsum("bhcd,bhde->bhce", k_n, W1)         # (b, h, c, m)
        hsil = F.silu(pre)
        pred = torch.einsum("bhce,bhed->bhcd", hsil, W2)       # (b, h, c, d)
        res = v_n - pred
        gW2 = self._norm_grad(-torch.einsum("bhce,bhcd->bhced", hsil, res))     # (b, h, c, m, d)
        herr = torch.einsum("bhed,bhcd->bhce", W2, res) * silu_grad(pre)  # (b, h, c, m)
        gW1 = self._norm_grad(-torch.einsum("bhcd,bhce->bhcde", k_n, herr))     # (b, h, c, d, m)

        # Momentum: mW_i = sum_{j<=i} coefM[i,j] gW_j + (prod nu) * mW_carry.
        mW1n = torch.einsum("bhij,bhjde->bhide", coefM, gW1) + An * mW1.unsqueeze(2)
        mW2n = torch.einsum("bhij,bhjed->bhied", coefM, gW2) + An * mW2.unsqueeze(2)
        # Weights: W_i = (prod alpha) * W_carry - sum_{j<=i} coefW[i,j] beta_j mW_j.
        bmW1 = beta_n[..., None, None] * mW1n                  # (b, h, c, d, m)
        bmW2 = beta_n[..., None, None] * mW2n
        W1i = self._ball(Bn * W1.unsqueeze(2) - torch.einsum("bhij,bhjde->bhide", coefW, bmW1), w1_home)
        W2i = self._ball(Bn * W2.unsqueeze(2) - torch.einsum("bhij,bhjed->bhied", coefW, bmW2), w2_home)

        # Read each token with its own updated memory.
        pre_q = torch.einsum("bhcd,bhcde->bhce", q_n, W1i)     # (b, h, c, m)
        out_n = torch.einsum("bhce,bhced->bhcd", F.silu(pre_q), W2i)
        return out_n, W1i[:, :, -1], W2i[:, :, -1], mW1n[:, :, -1], mW2n[:, :, -1]

    def _chunk_scan(self, q, k, v, beta, nu, alpha):
        """Chunked mini-batch scan: process `chunk_size` tokens at once with batched matmuls.

        The only approximation vs `_recurrent_scan` is that the inner-loop gradients within a chunk
        are taken at the *chunk-start* weights (the TTT/Titans "mini-batch gradient descent" form);
        the momentum + weight-decay recurrence is then applied exactly via cumulative-decay sums.
        With chunk_size=1 this is identical to the per-token loop. q/k/v: (b, h, l, d); gates (b, l, h).

        With `checkpoint_chunks` (default), each chunk step is gradient-checkpointed: autograd keeps
        only the chunk-boundary states and recomputes the step's per-token weight tensors in
        backward. Those tensors -- ~16 of (b, h, c, d, m) per chunk -- are what OOMed wide-memory
        points (hd32m4: ~13 GB at batch 256 on a 16 GB T4); checkpointed, the same forward holds
        ~1 GB for one extra scan's worth of recompute.
        """
        B, H, T, Dk = q.shape
        C = min(self.chunk_size, T)
        pad = (C - T % C) % C
        if pad:  # pad the end of the sequence so T is a multiple of C (sliced off at the end)
            q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))          # (b, h, l, d) -> pad l
            beta, nu, alpha = (F.pad(x, (0, 0, 0, pad)) for x in (beta, nu, alpha))  # (b, l, h) -> pad l
        Tp = q.shape[2]
        NT = Tp // C

        qc, kc, vc = (x.view(B, H, NT, C, Dk) for x in (q, k, v))            # (b, h, nt, c, d)
        # gates (b, l, h) -> (b, h, nt, c)
        betac, nuc, alphac = (x.view(B, NT, C, H).permute(0, 3, 1, 2) for x in (beta, nu, alpha))

        # Per-chunk cumulative log-decays (bounded: nu, alpha in (0, 1] -> cum <= 0, exp <= 1).
        cumnu = nuc.clamp_min(1e-6).log().cumsum(-1)    # (b, h, nt, c)
        cumal = alphac.clamp_min(1e-6).log().cumsum(-1)

        tril = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))  # lower incl diagonal

        def decay_coef(cum_n):  # coef[i, j] = exp(cum_i - cum_j) for j <= i else 0  (j>i masked pre-exp)
            diff = cum_n.unsqueeze(-1) - cum_n.unsqueeze(-2)        # (b, h, i, j)
            return diff.masked_fill(~tril, 0.0).exp().masked_fill(~tril, 0.0)

        # Carried memory state, reset to the learned init each sequence.
        W1 = self.mem_W1.float().expand(B, -1, -1, -1).clone()  # (b, h, d, m)
        W2 = self.mem_W2.float().expand(B, -1, -1, -1).clone()  # (b, h, m, d)
        mW1 = torch.zeros_like(W1)
        mW2 = torch.zeros_like(W2)
        w1_home, w2_home = self._home_norms(extra_dims=1)  # (1, h, 1, 1, 1) each; only used if weight_norm

        use_ckpt = self.checkpoint_chunks and torch.is_grad_enabled()
        outs = []
        for n in range(NT):
            args = (qc[:, :, n], kc[:, :, n], vc[:, :, n],           # (b, h, c, d)
                    betac[:, :, n],                                  # (b, h, c)
                    cumnu[:, :, n].exp()[..., None, None],           # (b, h, c, 1, 1) carry decay (momentum)
                    cumal[:, :, n].exp()[..., None, None],           # (b, h, c, 1, 1) carry decay (weights)
                    decay_coef(cumnu[:, :, n]), decay_coef(cumal[:, :, n]),
                    W1, W2, mW1, mW2, w1_home, w2_home)
            if use_ckpt:
                out_n, W1, W2, mW1, mW2 = checkpoint(self._chunk_step, *args, use_reentrant=False)
            else:
                out_n, W1, W2, mW1, mW2 = self._chunk_step(*args)
            outs.append(out_n)

        return torch.cat(outs, dim=2)[:, :, :T]

    def state_size(self, sequence_length: int = 2048) -> int:
        # Fast-weight memory per head: W1 (head_dim, mem_hidden) + W2 (mem_hidden, head_dim).
        # Independent of sequence length (the momentum buffers mW1/mW2 are omitted, as the small
        # conv state is omitted for the other mixers).
        return self.num_heads * 2 * self.head_dim * self.memory_hidden


def build(cfg, layer_idx: int) -> nn.Module:
    return Titans(
        cfg.d_model, num_heads=cfg.titans_num_heads, memory_mult=cfg.titans_memory_mult,
        head_dim=cfg.titans_head_dim, max_inner_lr=cfg.titans_max_inner_lr,
        conv_size=cfg.titans_conv_size, use_short_conv=cfg.titans_use_short_conv,
        mode=cfg.titans_mode, chunk_size=cfg.titans_chunk_size,
        update_norm=cfg.titans_update_norm, weight_norm=cfg.titans_weight_norm,
        update_eps=cfg.titans_update_eps, checkpoint_chunks=cfg.titans_checkpoint_chunks,
        layer_idx=layer_idx,
    )


def _titans_head_dim(cfg) -> int:
    """head_dim = titans_head_dim, or d_model // num_heads when decoupled knob is unset."""
    return cfg.titans_head_dim if cfg.titans_head_dim is not None else cfg.d_model // cfg.titans_num_heads


def state_size_bytes(cfg, n_layers: int, seq_len: int) -> int:
    """Closed form of LanguageModel.state_size for Titans:
    4 * n_layers * num_heads * 2 * head_dim * mem_hidden  (head_dim = titans_head_dim or
    d_model // num_heads, mem_hidden = memory_mult * head_dim). With a single head this is
    8 * memory_mult * n_layers * head_dim**2 -- independent of d_model.

    Independent of sequence length (bounded recurrent fast-weight state)."""
    head_dim = _titans_head_dim(cfg)
    mem_hidden = cfg.titans_memory_mult * head_dim
    return 4 * n_layers * cfg.titans_num_heads * 2 * head_dim * mem_hidden


def dims_str(cfg) -> str:
    head_dim = _titans_head_dim(cfg)
    mem_hidden = cfg.titans_memory_mult * head_dim
    return f"num_heads={cfg.titans_num_heads:>3d}  head_dim={head_dim:>3d}  mem_hidden={mem_hidden:>4d}"


SPEC = MixerSpec(
    name="titans",
    build=build,
    state_size_bytes=state_size_bytes,
    dims_str=dims_str,
)
