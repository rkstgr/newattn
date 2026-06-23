"""MQAR exp002 -- Mamba2: state size vs. recall accuracy.

The Mamba2 counterpart of exp001b: same MQAR task and harness, but the MHA mixer is
replaced by a pure-PyTorch Mamba2 SSM (no CUDA kernels -- runs on CPU or GPU). State size
= 4 * n_layers * (expand * d_model) * d_state bytes, *independent of sequence length*
(bounded recurrent state). Accuracy is therefore capped by state size and rises toward 1.0
as the state grows -- the headline zoology recall-vs-state curve for sub-quadratic mixers.

Run:
    python experiments/exp002_mamba2.py
    python experiments/exp002_mamba2.py --wandb-mode disabled
    python experiments/exp002_mamba2.py --help
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_D_MODELS, DEFAULT_LR_PER_D_MODEL, SweepConfig

DEFAULTS = SweepConfig(
    mixer="mamba2",
    exp_id="exp002",
    d_models=DEFAULT_D_MODELS["mamba2"],  # [8, 16, 32, 48, 64]
    lr_per_d_model=DEFAULT_LR_PER_D_MODEL["mamba2"],  # {8:3e-3, 16:2e-3, 32:1e-3, 48:1e-3, 64:8e-4}
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
