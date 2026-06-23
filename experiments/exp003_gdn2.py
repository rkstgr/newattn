"""MQAR exp003 -- Gated DeltaNet 2 (selectable mixer): state size vs. recall accuracy.

Extends exp002 with a selectable sub-quadratic mixer. The default is `gdn2`, a pure-PyTorch
port of fla's GatedDeltaNet2 (no Triton), whose decoupled erase/write delta-rule update is
generally more recall-efficient per state byte than a diagonal SSM. State size =
4 * n_layers * num_heads * head_dim * head_v_dim bytes (num_heads = d_model // head_dim),
independent of sequence length.

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
from newattn.config import DEFAULT_D_MODELS, DEFAULT_LR_PER_D_MODEL, SweepConfig

DEFAULTS = SweepConfig(
    mixer="gdn2",
    exp_id="exp003",
    d_models=DEFAULT_D_MODELS["gdn2"],  # [64, 128, 192, 256, 320]
    lr_per_d_model=DEFAULT_LR_PER_D_MODEL["gdn2"],
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
