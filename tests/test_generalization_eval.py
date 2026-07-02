"""Guards for the post-training generalization-eval machinery (exp006).

* `build_eval_dataloader` must be deterministic per (seed, setting) and — critically —
  independent of the *global* torch RNG state: `random_non_queries` draws distractors from the
  global generator, so without the internal fork_rng the eval set would depend on how much RNG
  state training consumed, and different runs would be evaluated on different data.
* `_build_model_config` must let `model_overrides={"max_position_embeddings": 0}` (NoPE) win over
  the task-derived default without a duplicate-kwarg TypeError.
* `run_generalization_evals` must skip (not crash on) settings longer than a model's learned
  position-embedding table, and run everything else.

Runs standalone (``python tests/test_generalization_eval.py``) or under pytest if installed.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newattn.config import EvalSetting, MQARTaskConfig, SweepConfig  # noqa: E402
from newattn.data import build_eval_dataloader  # noqa: E402
from newattn.model import LanguageModel  # noqa: E402
from newattn.sweep import _build_model_config, run_generalization_evals  # noqa: E402

TASK = MQARTaskConfig(vocab_size=256, input_seq_len=64, num_kv_pairs=8, num_test_examples=16)


def test_eval_dataloader_deterministic():
    """Same (task, seed) -> identical test set, no matter the global torch RNG state."""
    _, fp1 = build_eval_dataloader(TASK, seed=123, batch_size=8)
    torch.manual_seed(999)  # perturb the global generator like a training run would
    torch.randn(1000)
    _, fp2 = build_eval_dataloader(TASK, seed=123, batch_size=8)
    assert fp1 == fp2, "eval set depends on global torch RNG state (fork_rng regression)"

    import dataclasses
    _, fp3 = build_eval_dataloader(dataclasses.replace(TASK, num_kv_pairs=16), seed=123, batch_size=8)
    assert fp3 != fp1, "different settings should yield different test sets"
    _, fp4 = build_eval_dataloader(TASK, seed=124, batch_size=8)
    assert fp4 != fp1, "different seeds should yield different test sets"


def test_model_config_override_wins():
    """NoPE via model_overrides must beat the task-derived pos-emb default (and not TypeError)."""
    cfg = SweepConfig(mixer="attention", task=TASK)
    mc = _build_model_config(cfg, d_model=8, overrides={"max_position_embeddings": 0})
    assert mc.max_position_embeddings == 0
    mc = _build_model_config(cfg, d_model=8, overrides={})
    assert mc.max_position_embeddings == TASK.input_seq_len


def test_eval_skip_and_run():
    """Attention (learned pos-emb 32) runs the 32-cell, skips the 64-cell; NoPE mamba2 runs both."""
    torch.manual_seed(0)
    settings = [EvalSetting(32, 4, num_examples=32, batch_size=16),
                EvalSetting(64, 8, num_examples=32, batch_size=16)]
    base_task = MQARTaskConfig(vocab_size=128, input_seq_len=32, num_kv_pairs=4,
                               num_test_examples=32)
    common = dict(base_task=base_task, settings=settings, seed=0, device="cpu",
                  use_amp=False, amp_dtype="bfloat16", default_batch_size=16)

    attn_cfg = SweepConfig(mixer="attention", task=base_task)
    attn = LanguageModel(_build_model_config(attn_cfg, d_model=8, overrides={}))
    evals = run_generalization_evals(attn, **common)
    assert evals["s32_kv4"]["skipped"] is None and 0.0 <= evals["s32_kv4"]["accuracy"] <= 1.0
    assert evals["s64_kv8"]["skipped"] == "pos_emb" and evals["s64_kv8"]["accuracy"] is None

    mamba_cfg = SweepConfig(mixer="mamba2", task=base_task)
    mamba = LanguageModel(_build_model_config(mamba_cfg, d_model=8,
                                              overrides={"max_position_embeddings": 0, "d_state": 8}))
    evals = run_generalization_evals(mamba, **common)
    for label in ("s32_kv4", "s64_kv8"):
        assert evals[label]["skipped"] is None and 0.0 <= evals[label]["accuracy"] <= 1.0


if __name__ == "__main__":
    for fn in [test_eval_dataloader_deterministic, test_model_config_override_wins,
               test_eval_skip_and_run]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("All generalization-eval tests passed.")
