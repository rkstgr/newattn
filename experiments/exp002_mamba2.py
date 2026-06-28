"""MQAR exp002 -- Mamba2: state size vs. recall accuracy.

The Mamba2 counterpart of exp001b: same MQAR task and harness, but the MHA mixer is
replaced by a pure-PyTorch Mamba2 SSM (no CUDA kernels -- runs on CPU or GPU). State size
= 4 * n_layers * (expand * d_model) * d_state bytes, *independent of sequence length*
(bounded recurrent state). The state is swept via the SSM state dim `d_state` at a fixed
d_model=32, so accuracy is plotted against state size decoupled from the residual-stream width.

Run:
    python experiments/exp002_mamba2.py
    python experiments/exp002_mamba2.py --wandb-mode disabled
    python experiments/exp002_mamba2.py --help
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_POINTS, SweepConfig

DEFAULTS = SweepConfig(
    mixer="mamba2",
    exp_id="exp002",
    d_model=32,  # fixed residual-stream width; state size is swept via points
    points=DEFAULT_POINTS["mamba2"],  # d_state in {4, 8, 16, 32, 64, 128}
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
