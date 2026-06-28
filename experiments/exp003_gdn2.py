"""MQAR exp003 -- Gated DeltaNet 2 (selectable mixer): state size vs. recall accuracy.

Extends exp002 with a selectable sub-quadratic mixer. The default is `gdn2`, a pure-PyTorch
port of fla's GatedDeltaNet2 (no Triton), whose decoupled erase/write delta-rule update is
generally more recall-efficient per state byte than a diagonal SSM. State size =
4 * n_layers * num_heads * head_dim * head_v_dim bytes, independent of sequence length. The
state is swept via (gdn2_head_dim, gdn2_expand_v) at a fixed d_model=32 (single head), so the
x-axis is the state size decoupled from the residual-stream width.

The default `gdn2` runs anywhere (CPU or any GPU, incl. Turing/T4) -- no CUDA/Triton needed.
For the fast fla Triton kernels (Ampere+ GPU, bf16) pass `--mixer gdn2_triton` (needs
`pip install flash-linear-attention`). Pass `--mixer mamba2` for the Mamba2 overlay.

Run:
    python experiments/exp003_gdn2.py                          # pure-PyTorch gdn2 (CPU/T4)
    pixi run -e gdn2 python experiments/exp003_gdn2.py --mixer gdn2_triton   # fla kernels (GPU)
    python experiments/exp003_gdn2.py --mixer mamba2          # Mamba2 comparison
    python experiments/exp003_gdn2.py --help
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_POINTS, SweepConfig

DEFAULTS = SweepConfig(
    mixer="gdn2",
    exp_id="exp003",
    d_model=32,  # fixed residual-stream width; state size is swept via points
    points=DEFAULT_POINTS["gdn2"],  # (head_dim, expand_v): (8,1),(16,1),(16,2),(32,1),(32,2),(48,2),(64,2)
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
