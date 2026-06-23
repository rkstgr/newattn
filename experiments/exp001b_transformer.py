"""MQAR exp001b -- Transformer (attention): state size vs. recall accuracy.

Recreates the central Zoology experiment for a decoder-only transformer. State size = KV
cache = 4 * n_layers * 2 * d_model * seq_len bytes; with the task difficulty fixed,
`d_model` is the knob that moves us along the x-axis. Attention has effectively unbounded
state, so accuracy rises with width and saturates near 1.0.

Run:
    python experiments/exp001b_transformer.py                 # online W&B (prompts login)
    python experiments/exp001b_transformer.py --wandb-mode disabled
    python experiments/exp001b_transformer.py --help          # all overrides
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_D_MODELS, DEFAULT_LR_PER_D_MODEL, SweepConfig

DEFAULTS = SweepConfig(
    mixer="attention",
    exp_id="exp001b",
    ##d_models=DEFAULT_D_MODELS["attention"],  # [8, 16, 64, 128, 192]
    d_models=[32, 48], # post-fill these sizes
    lr_per_d_model=DEFAULT_LR_PER_D_MODEL["attention"],
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
