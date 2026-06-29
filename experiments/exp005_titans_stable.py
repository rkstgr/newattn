"""MQAR exp005 -- Titans neural memory with a stabilized inner loop (fixes the hd16m4 NaN).

Same model, sweep and state-size x-axis as exp004, but with the Titans test-time inner loop
made numerically stable so the wider-memory points no longer diverge. In exp004, `hd16m4`
(chunk 8) goes to NaN at ~step 1360 while `hd16m2` trains fine: the inner gradient-descent
memoriser is stable only when `beta * ||h||^2 < 2`, and `||h||^2` scales with the memory-MLP
hidden width `mem_hidden = memory_mult * head_dim`, so doubling `memory_mult` (m2 -> m4) roughly
doubles the inner-loop curvature and pushes the fixed inner LR past the stability boundary.

exp005 turns on two norm-constraints (off by default in exp004), following the recent
norm-constrained test-time-training literature:
  * `titans_update_norm="frobenius"` -- normalize the inner-loop gradient to a unit-norm update
    *direction*, so the step size is decoupled from ||grad|| (hence from ||h||^2 / state width).
    Same end as Muon in Atlas (arXiv:2505.23735) and LaCT (arXiv:2505.23884); the cheap Frobenius
    form is the "constant update norm + angular step" idea of Hyperball (arXiv:2603.28743). The
    underlying theory is the norm-ball LMO view of Muon/Scion (arXiv:2502.07529, arXiv:2506.15054).
  * `titans_weight_norm=True` -- cap each fast-weight matrix at its home (init) Frobenius norm
    (a no-op inside the ball), arresting the "magnitude explosion" of accumulating fast-weight
    updates that LaCT (arXiv:2505.23884) fixes with per-update weight L2-normalization.

Defaults to the previously-diverging configuration -- chunk mode, chunk size 8 -- so running this
file directly exercises exactly the case that NaN'd in exp004 and shows it now stays finite.

Run:
    python experiments/exp005_titans_stable.py                 # stabilized titans, chunk 8
    python experiments/exp005_titans_stable.py --mode recurrent # exact per-token loop (also stable)
    python experiments/exp005_titans_stable.py --help
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

from newattn.cli import run_experiment
from newattn.config import DEFAULT_POINTS, SweepConfig

DEFAULTS = SweepConfig(
    mixer="titans",
    exp_id="exp005",
    d_model=32,  # fixed residual-stream width; state size is swept via points
    points=DEFAULT_POINTS["titans"],  # (head_dim, memory_mult): (16,2),(16,4),(32,2),(32,4),(48,2),(64,2)
    # Stabilized inner loop + the chunk-8 scan that diverged in exp004 (applied to every point).
    model_overrides={
        "titans_mode": "chunk",
        "titans_chunk_size": 8,
        "titans_update_norm": "frobenius",
        "titans_weight_norm": True,
    },
    seed=123,
    wandb_project="zoology-mqar",
    wandb_entity=None,  # set to your W&B entity, or pass --wandb-entity / WANDB_ENTITY
    wandb_mode="online",
)

if __name__ == "__main__":
    run_experiment(DEFAULTS)
