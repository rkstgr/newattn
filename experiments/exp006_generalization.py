"""MQAR exp006 -- architecture generalization: train on the standard setting, evaluate harder/longer.

Compares the transformer (attention, d_model 32 -- the smallest configuration that solves MQAR at
seq 128 / 8 KV pairs, exp001b) against mamba2, gdn2 and titans. Two tiers per recurrent mixer:
a ~64k-byte state matched to the transformer's KV cache at the training length, and a uniform
8k small tier (7 runs total). All models train on the standard task (seq_len 128, 8 KV pairs,
vocab 8192); after training, each is evaluated in-process on an 11-cell grid of harder settings
-- more KV pairs (capacity stress) and longer sequences (retention stress).

Position embeddings: the recurrent mixers train with `max_position_embeddings=0` (NoPE -- their
short convolutions + recurrence carry position information), so they can run at any eval length.
The transformer keeps its learned absolute position embeddings (sized to the training length,
which cannot extrapolate) and records N/A on eval cells longer than 128 -- "cannot run" is kept
distinct from "runs and fails". Each eval cell uses a fresh deterministic test set shared across
all runs (verified by fingerprint). Weights + configs are saved under results/exp006/ and
uploaded to W&B, so runs can be re-evaluated later without retraining.

Run:
    python experiments/exp006_generalization.py                    # all 7 runs + combined figure
    python experiments/exp006_generalization.py --mixers gdn2      # one architecture (shard/rerun)
    python experiments/exp006_generalization.py --wandb-mode disabled --max-epochs 1 \
        --num-train-examples 512 --num-test-examples 128 --eval-num-examples 64   # smoke test
"""
import _bootstrap  # noqa: F401  (puts ./src on sys.path if newattn isn't installed)

import argparse
import copy
import json
import os
import uuid
from dataclasses import asdict, replace

from newattn.config import DEFAULT_POINTS, EvalSetting, SweepConfig
from newattn.sweep import MIXER_LABEL, run_sweep

EXP_ID = "exp006"
GROUP = f"mqar-generalization-{EXP_ID}"
OUT_DIR = os.path.join("results", EXP_ID)
NOPE = {"max_position_embeddings": 0}
TITANS_STABLE = {"titans_mode": "chunk", "titans_chunk_size": 8,  # exp005's validated config
                 "titans_update_norm": "frobenius", "titans_weight_norm": True}

# Eval grid: seq_len x num_kv_pairs, all cells satisfy the generator constraint 4*kv <= seq_len
# (the boundary cells 128/32, 256/64, 512/128 are maximally packed -- the hardest column of each
# row). s128_kv8 is a fresh draw of the *training* distribution (in-distribution control).
# Batch sizes shrink with length to cap activation memory.
EVAL_GRID = [
    EvalSetting(128, 8), EvalSetting(128, 16), EvalSetting(128, 32),
    EvalSetting(256, 8, batch_size=128), EvalSetting(256, 16, batch_size=128),
    EvalSetting(256, 32, batch_size=128), EvalSetting(256, 64, batch_size=128),
    EvalSetting(512, 8, batch_size=64), EvalSetting(512, 32, batch_size=64),
    EvalSetting(512, 64, batch_size=64), EvalSetting(512, 128, batch_size=64),
]
GRID_SEQ = sorted({s.input_seq_len for s in EVAL_GRID})
GRID_KV = sorted({s.num_kv_pairs for s in EVAL_GRID})
TRAIN_CELL = (128, 8)


def pick(mixer: str, *labels: str):
    pts = [replace(pt) for pt in DEFAULT_POINTS[mixer] if pt.label in labels]
    assert len(pts) == len(labels), f"unknown point label(s) for {mixer!r}: {labels}"
    return pts


def base(mixer: str, points, model_overrides: dict | None = None) -> SweepConfig:
    return SweepConfig(
        mixer=mixer, exp_id=EXP_ID, d_model=32, points=points,
        model_overrides=dict(model_overrides or {}), seed=123,
        eval_settings=list(EVAL_GRID), out_dir=OUT_DIR, group=GROUP,
        wandb_project="zoology-mqar", wandb_entity=None, wandb_mode="online",
    )


# 64k tier (matched to the transformer's KV cache @ seq 128) + 8k small tier per recurrent mixer.
DEFAULTS: list[SweepConfig] = [
    base("attention", pick("attention", "baseline")),                # 64k KV cache, learned pos-emb
    base("mamba2", pick("mamba2", "ds128", "ds16"), NOPE),           # 64k / 8k
    base("gdn2", pick("gdn2", "hd64v2", "hd32v1"), NOPE),            # 64k / 8k (first post-fix gdn2 run)
    base("titans", pick("titans", "hd32m4", "hd16m2"), {**NOPE, **TITANS_STABLE}),  # 64k / 8k
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MQAR exp006: cross-architecture generalization (train easy, eval harder/longer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mixers", default=None,
                   help="comma-separated subset of architectures to run, e.g. 'gdn2' or 'mamba2,titans'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--num-train-examples", type=int, default=None, help="handy for a quick smoke test")
    p.add_argument("--num-test-examples", type=int, default=None)
    p.add_argument("--eval-num-examples", type=int, default=None,
                   help="shrink every eval-grid cell to this many examples (smoke tests)")
    p.add_argument("--compile", dest="compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the model for training (evals always run the eager module)")
    # W&B (also read from WANDB_MODE / WANDB_ENTITY / WANDB_PROJECT env vars; CLI wins)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-project", default=None)
    return p.parse_args(argv)


def configs_from_args(args) -> list[SweepConfig]:
    cfgs = copy.deepcopy(DEFAULTS)
    if args.mixers:
        want = {m.strip() for m in args.mixers.split(",")}
        unknown = want - {c.mixer for c in cfgs}
        assert not unknown, f"unknown mixer(s) {sorted(unknown)}; available: {[c.mixer for c in cfgs]}"
        cfgs = [c for c in cfgs if c.mixer in want]
    for cfg in cfgs:
        if args.seed is not None:
            cfg.seed = args.seed
            cfg.train.seed = args.seed
        if args.max_epochs is not None:
            cfg.train.max_epochs = args.max_epochs
        if args.num_train_examples is not None:
            cfg.task.num_train_examples = args.num_train_examples
        if args.num_test_examples is not None:
            cfg.task.num_test_examples = args.num_test_examples
        if args.eval_num_examples is not None:
            cfg.eval_settings = [replace(s, num_examples=args.eval_num_examples)
                                 for s in cfg.eval_settings]
        if args.compile is not None:
            cfg.train.compile = args.compile
        cfg.wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or cfg.wandb_mode
        cfg.wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or cfg.wandb_project
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        if entity is not None:
            cfg.wandb_entity = entity
    return cfgs


def check_eval_fingerprints(runs: list[dict]):
    """Every run must have been evaluated on byte-identical test sets, cell by cell."""
    by_label: dict[str, set] = {}
    for r in runs:
        for lbl, rec in r["evals"].items():
            if rec["fingerprint"] is not None:
                by_label.setdefault(lbl, set()).add(rec["fingerprint"])
    mismatched = {lbl: sorted(fps) for lbl, fps in by_label.items() if len(fps) > 1}
    assert not mismatched, f"eval sets differ across runs (determinism regression): {mismatched}"


def plot_generalization(runs: list[dict], out_path: str):
    """2 rows x 4 mixers of accuracy heatmaps over the (seq_len x kv) grid.

    Top row = 64k-state tier, bottom = 8k tier. Hatched cells = the model cannot run there
    (learned pos-emb shorter than the sequence); light-gray = not in the eval grid; the black
    box marks the training distribution."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    mixers = [m for m in ("attention", "mamba2", "gdn2", "titans") if any(r["mixer"] == m for r in runs)]
    tiers = []  # rows: largest state first
    for m in mixers:
        tiers.append(sorted((r for r in runs if r["mixer"] == m),
                            key=lambda r: -r["state_size"]))
    n_rows = max(len(t) for t in tiers)

    # no sharex/sharey: the hidden attention slot would otherwise swallow its row's tick labels
    fig, axes = plt.subplots(n_rows, len(mixers), figsize=(3.6 * len(mixers), 2.6 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")

    im = None
    for col, (mixer, tier) in enumerate(zip(mixers, tiers)):
        for row in range(n_rows):
            ax = axes[row][col]
            if row >= len(tier):
                ax.axis("off")
                ax.text(0.5, 0.5, "n/a -- attention state\nis unbounded (KV cache)",
                        ha="center", va="center", fontsize=8, color="gray", transform=ax.transAxes)
                continue
            run = tier[row]
            acc = np.full((len(GRID_SEQ), len(GRID_KV)), np.nan)
            skipped = np.zeros(acc.shape, dtype=bool)
            for s in EVAL_GRID:
                i, j = GRID_SEQ.index(s.input_seq_len), GRID_KV.index(s.num_kv_pairs)
                rec = run["evals"].get(s.label)
                if rec is None:
                    continue
                if rec["skipped"]:
                    skipped[i, j] = True
                else:
                    acc[i, j] = rec["accuracy"]

            im = ax.imshow(np.ma.masked_invalid(acc), vmin=0, vmax=1, cmap=cmap, aspect="auto")
            in_grid = {(GRID_SEQ.index(s.input_seq_len), GRID_KV.index(s.num_kv_pairs))
                       for s in EVAL_GRID}
            for i in range(len(GRID_SEQ)):
                for j in range(len(GRID_KV)):
                    if skipped[i, j]:
                        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, hatch="///",
                                               fill=False, edgecolor="gray", linewidth=0))
                        ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="gray")
                    elif (i, j) not in in_grid:
                        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor="0.94", edgecolor="none"))
                    else:
                        ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ti, tj = GRID_SEQ.index(TRAIN_CELL[0]), GRID_KV.index(TRAIN_CELL[1])
            ax.add_patch(Rectangle((tj - .5, ti - .5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=1.6))
            ax.set_title(f"{MIXER_LABEL.get(mixer, mixer)}\n{run['label']} -- "
                         f"{run['state_size'] // 1024} KiB state", fontsize=9)
            ax.set_xticks(range(len(GRID_KV)), GRID_KV)
            ax.set_yticks(range(len(GRID_SEQ)), GRID_SEQ)
            if row == len(tier) - 1:  # visually-bottom panel of this column
                ax.set_xlabel("num KV pairs")
            if col == min(c for c, t in enumerate(tiers) if len(t) > row):  # leftmost visible in row
                ax.set_ylabel("eval seq_len")

    fig.suptitle("MQAR generalization: trained on seq 128 / 8 KV pairs (vocab 8192), "
                 "black box = training distribution", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="recall accuracy")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    return fig


def log_wandb_summary(runs: list[dict], fig, *, wandb_mode: str, project: str, entity: str | None):
    if wandb_mode == "disabled":
        return
    import wandb

    summary = wandb.init(project=project, entity=entity, name=f"{EXP_ID}-summary-{uuid.uuid4().hex[:8]}",
                         group=GROUP, job_type="summary", mode=wandb_mode, reinit=True,
                         tags=["mqar", "generalization", "summary"])
    columns = ["mixer", "run_label", "state_size", "eval_label", "seq_len", "num_kv_pairs",
               "accuracy", "loss", "skipped"]
    data = [[r["mixer"], r["label"], r["state_size"], s.label, s.input_seq_len, s.num_kv_pairs,
             rec["accuracy"], rec["loss"], rec["skipped"]]
            for r in runs for s in EVAL_GRID if (rec := r["evals"].get(s.label)) is not None]
    summary.log({"generalization_plot": wandb.Image(fig),
                 "generalization_results": wandb.Table(columns=columns, data=data)})
    summary.finish()
    print("Logged combined generalization plot + table to Weights & Biases.")


def main(argv=None):
    args = parse_args(argv)
    cfgs = configs_from_args(args)

    all_runs = []
    for cfg in cfgs:
        print(f"\n===== exp006 [{cfg.mixer}] =====")
        results = run_sweep(cfg, plot=False)  # one combined figure below instead of 4 per-mixer plots
        for r in results:
            r["mixer"] = cfg.mixer
        all_runs.extend(results)

    check_eval_fingerprints(all_runs)

    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({"exp_id": EXP_ID, "seed": cfgs[0].seed, "train_task": asdict(cfgs[0].task),
                   "eval_grid": [asdict(s) for s in cfgs[0].eval_settings], "runs": all_runs},
                  f, indent=2)
    print(f"\nWrote {results_path}")

    fig = plot_generalization(all_runs, out_path=os.path.join(OUT_DIR, f"{EXP_ID}_generalization.png"))
    log_wandb_summary(all_runs, fig, wandb_mode=cfgs[0].wandb_mode,
                      project=cfgs[0].wandb_project, entity=cfgs[0].wandb_entity)

    print("\nexp006 complete.")
    for r in all_runs:
        cells = "  ".join(f"{s.label}={rec['accuracy']:.3f}" if rec["accuracy"] is not None
                          else f"{s.label}=N/A"
                          for s in EVAL_GRID if (rec := r["evals"].get(s.label)) is not None)
        print(f"  {r['mixer']:<9s} {r['label']:<9s} state={r['state_size']:>7d}  "
              f"train_best={r['valid_best_accuracy']:.3f}\n    {cells}")
    return all_runs


if __name__ == "__main__":
    main()
