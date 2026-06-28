"""Training loop (based on zoology/train.py).

AdamW, cross-entropy with `ignore_index=-100` (so only answer positions count), and
per-example recall accuracy over non-ignored positions. On top of the zoology defaults:

* **Per-step LR schedule:** linear warmup over `warmup_epochs`, then cosine decay to 0.
  The peak LR is each sweep point's `SweepPoint.lr`.
* **Gradient clipping** at `max_norm=grad_clip`.
* **Early stopping** when the task is solved (`valid/accuracy > threshold`) or when
  validation accuracy plateaus for `patience` epochs. We report the **best** accuracy.
* **No weight decay on 1-D params** (norms, biases, and SSM `A_log`/`dt_bias`/`D`) --
  standard practice; decaying those would distort learned timescales.
* **Mixed precision** (optional, `use_amp`): the forward runs under bf16 autocast (the fla
  Triton kernels expect bf16/fp16); params stay fp32 and the loss is computed in fp32.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm.auto import tqdm

from .config import TrainParams

# Autocast dtype names -> torch dtype. T4/Turing has no native bf16; use "fp16".
_AMP_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def compute_metrics(preds, targets, ignore_index=-100):
    """Per-example recall accuracy over non-ignored positions (zoology.train.compute_metrics)."""
    accs = []
    for pred, target in zip(preds, targets):
        mask = target != ignore_index
        accs.append((pred[mask] == target[mask]).float().mean().item())
    return accs


def train_one_run(model, train_dl, test_dl, logger, peak_lr: float, train: TrainParams,
                  device: str, use_amp: bool = False):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()  # default ignore_index = -100

    amp_dtype = _AMP_DTYPES[train.amp_dtype]

    def forward_logits(inputs):
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            return model(inputs)

    # No weight decay on 1-D params (A_log, dt_bias, D, norm/bias) -- standard practice.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr,
    )

    # Per-step schedule: linear warmup over `warmup_epochs`, then cosine decay to 0.
    steps_per_epoch = len(train_dl)
    warmup_steps = max(1, int(train.warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, train.max_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:  # linear 0 -> 1
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # cosine 1 -> 0
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc, epochs_no_improve = 0.0, 0
    final_metrics = {}
    for epoch in range(train.max_epochs):
        # ---- train ----
        model.train()
        for inputs, targets in tqdm(train_dl, desc=f"train {epoch + 1}/{train.max_epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = forward_logits(inputs)
            loss = loss_fn(rearrange(logits, "... c -> (...) c").float(), targets.flatten())
            loss.backward()
            if train.grad_clip and train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
            optimizer.step()
            scheduler.step()
            logger.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch})

        # ---- eval ----
        model.eval()
        test_loss, accs = 0.0, []
        with torch.no_grad():
            for inputs, targets in test_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = forward_logits(inputs)
                loss = loss_fn(rearrange(logits, "... c -> (...) c").float(), targets.flatten())
                test_loss += loss.item() / len(test_dl)
                accs.extend(compute_metrics(logits.argmax(-1).cpu(), targets.cpu()))

        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, epochs_no_improve = acc, 0
        else:
            epochs_no_improve += 1
        final_metrics = {"valid/loss": test_loss, "valid/accuracy": acc, "valid/best_accuracy": best_acc}
        logger.log({"epoch": epoch, **final_metrics})
        print(f"  epoch {epoch + 1:>3}/{train.max_epochs}  valid/loss={test_loss:.4f}  "
              f"valid/accuracy={acc:.4f}  best={best_acc:.4f}  no_improve={epochs_no_improve}")

        # ---- early stopping: solved, or no val-accuracy improvement for `patience` epochs ----
        if acc > train.early_stopping_threshold:
            print(f"  early stop (solved): valid/accuracy {acc:.4f} > {train.early_stopping_threshold}")
            break
        if epochs_no_improve >= train.patience:
            print(f"  early stop (plateau): no valid/accuracy improvement for {train.patience} "
                  f"epochs (best={best_acc:.4f})")
            break

    final_metrics["valid/best_accuracy"] = best_acc
    return final_metrics
