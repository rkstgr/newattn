"""MQAR exp004 -- Titans neural memory (selectable mixer): state size vs. recall accuracy.

Like exp003, but the default mixer is `titans`: a pure-PyTorch port of the Titans neural
memory, whose per-head fast memory is a two-layer MLP updated at test time by an inner-loop
gradient step with momentum and weight decay (instead of the linear delta-rule recurrence of
gdn2/mamba2). State size = 4 * n_layers * num_heads * 2 * head_dim * mem_hidden bytes
(single head, mem_hidden = memory_mult * head_dim), independent of sequence length. The state
is swept via (titans_head_dim, titans_memory_mult) at a fixed d_model=32, so the x-axis is the
state size decoupled from the residual-stream width.

Pure-PyTorch -- runs anywhere (CPU or any GPU, incl. Turing/T4), no CUDA/Triton needed. Pass
`--mixer gdn2` / `--mixer mamba2` to compare against the linear-attention overlays.

Run:
    python experiments/exp004_titans.py                       # pure-PyTorch titans (CPU/T4)
    python experiments/exp004_titans.py --mixer gdn2          # Gated DeltaNet 2 comparison
    python experiments/exp004_titans.py --mixer mamba2        # Mamba2 comparison
    python experiments/exp004_titans.py --help
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_POINTS, SweepConfig

DEFAULTS = SweepConfig(
    mixer="titans",
    exp_id="exp004",
    d_model=32,  # fixed residual-stream width; state size is swept via points
    points=DEFAULT_POINTS["titans"],  # (head_dim, memory_mult): (16,2),(16,4),(32,2),(32,4),(48,2),(64,2)
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
