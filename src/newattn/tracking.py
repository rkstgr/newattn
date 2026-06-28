"""Weights & Biases harness (mirrors zoology.logger).

`WandbLogger` no-ops if W&B is disabled; `build_full_config` assembles the complete,
reproducible nested config (task, model, training, derived state size, seed, and the
runtime/library environment) that gets stored in W&B.
"""
from __future__ import annotations

import platform
from dataclasses import asdict

import numpy as np
import torch

from .config import ModelConfig, MQARTaskConfig, SweepPoint, TrainParams


def maybe_login(wandb_mode: str) -> str:
    """Log in unless disabled/offline. Returns the (possibly downgraded) mode."""
    if wandb_mode in ("disabled", "offline"):
        return wandb_mode
    import wandb

    try:
        wandb.login()
    except Exception as e:  # pragma: no cover - network/credentials dependent
        print(f"wandb.login() failed ({e}); falling back to offline mode.")
        return "offline"
    return wandb_mode


def build_full_config(*, mixer: str, task: MQARTaskConfig, train: TrainParams,
                      model_cfg: ModelConfig, points: list[SweepPoint], d_model: int,
                      sweep_id: str, seed: int, state_size: int, num_parameters: int,
                      peak_lr: float, fingerprint: str, device: str) -> dict:
    """The full, reproducible config saved to W&B."""
    return {
        "architecture": mixer,
        "task": asdict(task),
        "model": asdict(model_cfg),
        "train": asdict(train),
        "sweep": {
            "sweep_id": sweep_id,
            "mixer": mixer,
            "d_model": d_model,
            "n_points": len(points),
            "points": [asdict(pt) for pt in points],
        },
        "derived": {
            "state_size": state_size,
            "state_size_seq_len": task.input_seq_len,
            "num_parameters": num_parameters,
            "peak_learning_rate": peak_lr,
        },
        "reproducibility": {
            "seed": seed,
            "dataset_fingerprint": fingerprint,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic": True,
        },
    }


class WandbLogger:
    """Thin logger mirroring zoology.logger.WandbLogger (no-ops if W&B is disabled)."""

    def __init__(self, run):
        self.run = run

    def log(self, metrics: dict):
        if self.run is not None:
            self.run.log(metrics)
