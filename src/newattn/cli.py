"""Shared command-line entry point for the experiment scripts.

`run_experiment(defaults)` takes a `SweepConfig` of defaults, applies environment-variable
and command-line overrides (so Colab users can tweak knobs without editing files), then
runs the sweep. Run any experiment with `--help` to see the available flags.
"""
from __future__ import annotations

import argparse
import copy
import os

import dataclasses

from .config import DEFAULT_POINTS, SweepConfig
from .sweep import run_sweep


def _build_parser(defaults: SweepConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"MQAR state-size sweep ({defaults.exp_id}, default mixer={defaults.mixer}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mixer", choices=["attention", "mamba2", "gdn2", "gdn2_triton", "titans"], default=None,
                   help="sequence mixer (gdn2 = pure-PyTorch, gdn2_triton = fla kernels, "
                        "titans = MLP neural-memory; overrides the experiment default)")
    p.add_argument("--d-model", type=int, default=None,
                   help="fixed residual-stream width for the whole sweep (state size is swept via points)")
    p.add_argument("--lr", type=float, default=None,
                   help="flat peak LR for every point (overrides each point's lr)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="autocast dtype for AMP mixers like gdn2 (fp16 for Turing/T4; default bf16)")
    p.add_argument("--compile", dest="compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the model (Inductor + CUDA graphs); big speedup, esp. for titans")
    p.add_argument("--mode", choices=["recurrent", "chunk"], default=None,
                   help="titans inner-loop scan: 'recurrent' (exact per-token) or 'chunk' (faster mini-batch)")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="chunk length when --mode=chunk (smaller = closer to exact)")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--num-train-examples", type=int, default=None, help="handy for a quick smoke test")
    p.add_argument("--num-test-examples", type=int, default=None)
    # W&B (also read from WANDB_MODE / WANDB_ENTITY / WANDB_PROJECT env vars; CLI wins)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-project", default=None)
    return p


def config_from_args(defaults: SweepConfig, argv=None) -> SweepConfig:
    args = _build_parser(defaults).parse_args(argv)
    cfg = copy.deepcopy(defaults)

    if args.mixer:
        cfg.mixer = args.mixer
    if args.mixer and args.mixer != defaults.mixer:
        # switched mixer -> adopt that mixer's default sweep points (unless overridden below)
        cfg.points = [dataclasses.replace(pt) for pt in DEFAULT_POINTS[args.mixer]]
    if args.d_model is not None:
        cfg.d_model = args.d_model

    # --lr sets a flat LR for every point in the (possibly overridden) sweep
    if args.lr is not None:
        cfg.points = [dataclasses.replace(pt, lr=args.lr) for pt in cfg.points]

    if args.seed is not None:
        cfg.seed = args.seed
        cfg.train.seed = args.seed
    if args.amp_dtype is not None:
        cfg.train.amp_dtype = args.amp_dtype
    if args.compile is not None:
        cfg.train.compile = args.compile
    # titans chunked-scan knobs apply to every point (merged under each point's ModelConfig)
    if args.mode is not None:
        cfg.model_overrides["titans_mode"] = args.mode
    if args.chunk_size is not None:
        cfg.model_overrides["titans_chunk_size"] = args.chunk_size
    if args.max_epochs is not None:
        cfg.train.max_epochs = args.max_epochs
    if args.num_train_examples is not None:
        cfg.task.num_train_examples = args.num_train_examples
    if args.num_test_examples is not None:
        cfg.task.num_test_examples = args.num_test_examples

    # W&B: CLI flag > env var > experiment default
    cfg.wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or cfg.wandb_mode
    cfg.wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or cfg.wandb_project
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity is not None:
        cfg.wandb_entity = entity

    return cfg


def run_experiment(defaults: SweepConfig, argv=None):
    run_sweep(config_from_args(defaults, argv))
