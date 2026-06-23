"""MQAR exp003 -- Gated DeltaNet 2 (selectable mixer): state size vs. recall accuracy.

Extends exp002 with a selectable sub-quadratic mixer. The default is `gdn2`
(fla.layers.GatedDeltaNet2 from flash-linear-attention), whose delta-rule update is
generally more recall-efficient per state byte than a diagonal SSM. State size =
4 * n_layers * num_heads * head_dim * head_v_dim bytes (num_heads = d_model // head_dim),
independent of sequence length.

`gdn2` requires a CUDA GPU + Triton (and `pip install flash-linear-attention`); the
forward runs under bf16 autocast. Pass `--mixer mamba2` to run the CPU-friendly Mamba2
mixer instead (same harness, for an apples-to-apples overlay).

Run (needs a GPU for gdn2):
    pixi run -e gdn2 python experiments/exp003_gdn2.py
    python experiments/exp003_gdn2.py --mixer mamba2          # CPU-friendly comparison
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
