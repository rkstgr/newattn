"""MQAR exp001b -- Transformer (attention): state size vs. recall accuracy.

Recreates the central Zoology experiment for a decoder-only transformer. State size = KV
cache = 4 * n_layers * 2 * d_model * seq_len bytes. Attention has effectively unbounded state
(it grows with sequence length, not a bounded recurrent knob), so it serves as a single fixed
baseline at d_model=32 -- the reference accuracy that the sub-quadratic mixers are compared to.

Run:
    python experiments/exp001b_transformer.py                 # online W&B (prompts login)
    python experiments/exp001b_transformer.py --wandb-mode disabled
    python experiments/exp001b_transformer.py --help          # all overrides
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_POINTS, SweepConfig

DEFAULTS = SweepConfig(
    mixer="attention",
    exp_id="exp001b",
    d_model=32,  # fixed residual-stream width
    points=DEFAULT_POINTS["attention"],  # single baseline point (attention has no bounded state knob)
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
