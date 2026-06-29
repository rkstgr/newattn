"""Equivalence guards for the chunked sequence-mixer scans.

* Titans: the chunked mini-batch scan with ``chunk_size=1`` must reproduce the exact per-token
  loop (the only approximation in chunk mode is taking the inner-loop gradients at the chunk-start
  weights -- with one token per chunk there is nothing to approximate). Larger chunks are a
  speed/accuracy trade-off and are only checked for finiteness here.
* gdn2: the chunked WY scan must match the token-by-token reference (the trustworthy ground truth)
  -- this pins the chunk math as correct (it agrees to ~1e-6) and guards against regressions.

Runs standalone (``python tests/test_titans_chunk.py``) or under pytest if installed.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newattn.mixers.titans import Titans  # noqa: E402
from newattn.mixers.gdn2 import GatedDeltaNet2Naive  # noqa: E402

CONFIGS = [(16, 2, True), (16, 4, True), (32, 2, True), (64, 2, False)]


def test_titans_chunk1_equals_recurrent():
    """chunk_size=1 is bit-for-bit (to fp32 tol) the per-token loop, across configs."""
    torch.manual_seed(0)
    for head_dim, mm, conv in CONFIGS:
        m = Titans(d_model=32, num_heads=1, memory_mult=mm, head_dim=head_dim, use_short_conv=conv)
        x = torch.randn(4, 128, 32)
        m.mode = "recurrent"
        o_ref = m(x)
        m.mode, m.chunk_size = "chunk", 1
        o_chunk1 = m(x)
        max_diff = (o_ref - o_chunk1).abs().max().item()
        assert max_diff < 1e-4, f"hd={head_dim} mm={mm}: chunk1 vs loop diff {max_diff:.2e}"


def test_titans_chunk_finite():
    """Larger chunks (the mini-batch approximation) stay finite in forward and backward."""
    torch.manual_seed(0)
    for head_dim, mm, conv in CONFIGS:
        m = Titans(d_model=32, num_heads=1, memory_mult=mm, head_dim=head_dim, use_short_conv=conv)
        x = torch.randn(4, 128, 32, requires_grad=True)
        m.mode, m.chunk_size = "chunk", 64
        o = m(x)
        assert torch.isfinite(o).all(), f"hd={head_dim} mm={mm}: non-finite chunk output"
        g = torch.autograd.grad(o.sum(), x)[0]
        assert torch.isfinite(g).all(), f"hd={head_dim} mm={mm}: non-finite chunk grad"


def test_titans_stable_chunk1_equals_recurrent():
    """The stabilized inner loop (exp005: gradient-norm + weight-ball) still satisfies chunk1==loop."""
    torch.manual_seed(0)
    for head_dim, mm in [(16, 2), (16, 4), (32, 4)]:
        m = Titans(d_model=32, num_heads=1, memory_mult=mm, head_dim=head_dim,
                   update_norm="frobenius", weight_norm=True)
        x = torch.randn(4, 128, 32)
        m.mode = "recurrent"
        o_ref = m(x)
        m.mode, m.chunk_size = "chunk", 1
        max_diff = (o_ref - m(x)).abs().max().item()
        assert max_diff < 1e-4, f"stable hd={head_dim} mm={mm}: chunk1 vs loop diff {max_diff:.2e}"


def test_titans_stable_finite_in_blowup_regime():
    """In the high-curvature regime (||h||^2 >> 1) where the wide-memory baseline diverges to NaN,
    the exp005 stabilization (frobenius update-norm + weight ball) keeps fwd+bwd finite.

    Reproduces the hd16m4 failure mode directly by scaling the fast-weight init so beta*||h||^2 >> 2
    from the first token, instead of waiting ~1360 training steps for the projections to grow."""
    torch.manual_seed(0)

    def build(stabilize):
        kw = dict(update_norm="frobenius", weight_norm=True) if stabilize else {}
        m = Titans(d_model=32, num_heads=1, memory_mult=4, head_dim=16,
                   mode="chunk", chunk_size=8, **kw)
        with torch.no_grad():  # push the memory MLP into the divergent high-curvature regime
            m.mem_W1 *= 6.0
            m.mem_W2 *= 6.0
        return m

    x = torch.randn(4, 128, 32)
    base = build(False)(x)
    assert not torch.isfinite(base).all(), "baseline should blow up here (else the test no longer exercises the bug)"

    xs = x.clone().requires_grad_(True)
    out = build(True)(xs)
    assert torch.isfinite(out).all(), "stabilized chunk output should stay finite"
    g = torch.autograd.grad(out.sum(), xs)[0]
    assert torch.isfinite(g).all(), "stabilized chunk grad should stay finite"


def test_titans_chunk_odd_length():
    """Sequence length not divisible by chunk_size (pad path) preserves length and the chunk1 match."""
    torch.manual_seed(0)
    m = Titans(d_model=32, num_heads=1, memory_mult=2, head_dim=16)
    x = torch.randn(2, 130, 32)
    m.mode = "recurrent"
    o_ref = m(x)
    m.mode, m.chunk_size = "chunk", 64  # 130 = 2*64 + 2 -> padded
    o = m(x)
    assert o.shape == o_ref.shape, f"shape {tuple(o.shape)} != {tuple(o_ref.shape)}"
    assert torch.isfinite(o).all()
    m.chunk_size = 1
    assert (o_ref - m(x)).abs().max().item() < 1e-4


def test_gdn2_chunk_matches_recurrent():
    """gdn2's chunked WY scan reproduces the token-by-token ground truth (chunk math is correct)."""
    torch.manual_seed(0)
    for mode_pair in [(8, 1.0), (16, 1.0), (16, 2.0)]:  # (head_dim, expand_v)
        hd, ev = mode_pair
        x = torch.randn(3, 128, 32)
        common = dict(d_model=32, head_dim=hd, num_heads=1, expand_v=ev)
        rec = GatedDeltaNet2Naive(mode="fused_recurrent", **common)
        chk = GatedDeltaNet2Naive(mode="chunk", **common)
        chk.load_state_dict(rec.state_dict())  # identical weights, only the scan differs
        diff = (rec(x) - chk(x)).abs().max().item()
        assert diff < 1e-3, f"gdn2 hd={hd} ev={ev}: chunk vs recurrent diff {diff:.2e}"


if __name__ == "__main__":
    for fn in [test_titans_chunk1_equals_recurrent, test_titans_chunk_finite,
               test_titans_stable_chunk1_equals_recurrent, test_titans_stable_finite_in_blowup_regime,
               test_titans_chunk_odd_length, test_gdn2_chunk_matches_recurrent]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("All chunk-equivalence tests passed.")
