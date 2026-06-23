"""Shared command-line entry point for the experiment scripts.

`run_experiment(defaults)` takes a `SweepConfig` of defaults, applies environment-variable
and command-line overrides (so Colab users can tweak knobs without editing files), then
runs the sweep. Run any experiment with `--help` to see the available flags.
"""
from __future__ import annotations

import argparse
import copy
import os

from .config import DEFAULT_D_MODELS, DEFAULT_LR_PER_D_MODEL, SweepConfig
from .sweep import run_sweep


def _build_parser(defaults: SweepConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"MQAR state-size sweep ({defaults.exp_id}, default mixer={defaults.mixer}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mixer", choices=["attention", "mamba2", "gdn2"], default=None,
                   help="sequence mixer (overrides the experiment default)")
    p.add_argument("--d-models", type=int, nargs="+", default=None,
                   help="explicit width sweep (defaults are mixer-specific)")
    p.add_argument("--lr", type=float, default=None,
                   help="flat peak LR for every width (overrides the per-d_model map)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="autocast dtype for AMP mixers like gdn2 (fp16 for Turing/T4; default bf16)")
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
        # switched mixer -> adopt that mixer's default widths + LR map (unless overridden below)
        cfg.d_models = list(DEFAULT_D_MODELS[args.mixer])
        cfg.lr_per_d_model = dict(DEFAULT_LR_PER_D_MODEL[args.mixer])
    if args.d_models is not None:
        cfg.d_models = args.d_models

    # --lr sets a flat LR for the whole (possibly overridden) sweep
    if args.lr is not None:
        cfg.lr_per_d_model = {d: args.lr for d in cfg.d_models}

    if args.seed is not None:
        cfg.seed = args.seed
        cfg.train.seed = args.seed
    if args.amp_dtype is not None:
        cfg.train.amp_dtype = args.amp_dtype
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
