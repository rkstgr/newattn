"""Run a state-size-vs-recall sweep and plot the recreated zoology curve.

`run_sweep(SweepConfig)` trains one model per `SweepPoint` in `cfg.points` (all at the fixed
`cfg.d_model`), maps each point to its state size on the x-axis, logs everything to W&B, and
produces the final state-size-vs-accuracy figure (saved to disk and logged as a W&B image /
table / line).
"""
from __future__ import annotations

import json
import uuid

from .config import ModelConfig, SweepConfig
from .data import build_dataloaders
from .determinism import get_device, set_determinism
from .mixers import get_spec
from .model import LanguageModel
from .tracking import WandbLogger, build_full_config, maybe_login
from .train import train_one_run

MIXER_LABEL = {"attention": "Transformer (attention)", "mamba2": "Mamba2",
               "gdn2": "Gated DeltaNet 2", "gdn2_triton": "Gated DeltaNet 2 (Triton)",
               "titans": "Titans (neural memory)"}
MIXER_COLOR = {"attention": "#3b76af", "mamba2": "#c4694b", "gdn2": "#4b78c4",
               "gdn2_triton": "#6a4bc4", "titans": "#4bb37a"}


def _point_label(pt) -> str:
    """Compact run/plot label derived from a SweepPoint's overrides (fallback when no label set)."""
    if not pt.overrides:
        return "baseline"
    parts = []
    for k, v in pt.overrides.items():
        short = k.split("_")[-1]  # e.g. gdn2_head_dim -> dim, d_state -> state
        parts.append(f"{short}{v:g}" if isinstance(v, (int, float)) else f"{short}{v}")
    return "-".join(parts)


def _point_dims(cfg, pt) -> tuple[int, dict]:
    """Merge base + point overrides and split off d_model (a point may override the width too).
    Returns (d_model, remaining_overrides)."""
    overrides = {**cfg.model_overrides, **pt.overrides}
    d_model = overrides.pop("d_model", cfg.d_model)
    return d_model, overrides


def run_sweep(cfg: SweepConfig) -> list[dict]:
    device = get_device()
    spec = get_spec(cfg.mixer)
    sweep_id = uuid.uuid4().hex[:8]
    group = cfg.resolved_group()

    print(f"Using device: {device}  (mixer={cfg.mixer!r})")
    if device == "cpu":
        print("WARNING: no GPU detected -- runs will be slow. In Colab: Runtime > Change runtime type > GPU.")
    if spec.requires_cuda and device == "cpu":
        print(f"WARNING: mixer {cfg.mixer!r} needs a CUDA GPU (Triton kernels); on CPU the run will fail. "
              f"Switch the Colab runtime to GPU, or pick a CPU-friendly mixer.")

    # Planning table: each point -> a state size (the x-axis) and its peak LR, at fixed d_model.
    n_layers = ModelConfig().n_layers
    print(f"Planned sweep (mixer={cfg.mixer!r}; d_model={cfg.d_model}; state size is the x-axis):")
    for pt in cfg.points:
        d_model, overrides = _point_dims(cfg, pt)
        mc = ModelConfig(d_model=d_model, mixer=cfg.mixer, **overrides)
        state = spec.state_size_bytes(mc, n_layers, cfg.task.input_seq_len)
        label = pt.label or _point_label(pt)
        print(f"  {label:<12s}  d_model={d_model:>4d}  {spec.dims_str(mc)}  ->  "
              f"state_size={state:>11d}  ->  lr={pt.lr:.2e}")

    wandb_mode = maybe_login(cfg.wandb_mode)
    use_amp = spec.use_amp and device == "cuda"

    results = []
    for i, pt in enumerate(cfg.points):
        peak_lr = pt.lr
        label = pt.label or _point_label(pt)
        d_model, overrides = _point_dims(cfg, pt)
        print(f"\n=== Run {i + 1}/{len(cfg.points)}: {label} (d_model={d_model}, lr={peak_lr:.2e}) ===")
        set_determinism(cfg.seed)

        # data (identical task across runs) + model
        train_dl, test_dl, fingerprint = build_dataloaders(
            cfg.task, seed=cfg.seed, batch_size=cfg.train.batch_size,
            test_batch_size=cfg.train.test_batch_size)
        model_cfg = ModelConfig(d_model=d_model, mixer=cfg.mixer, vocab_size=cfg.task.vocab_size,
                                max_position_embeddings=cfg.task.input_seq_len, **overrides)
        model = LanguageModel(model_cfg)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_size = model.state_size(sequence_length=cfg.task.input_seq_len)
        full_config = build_full_config(
            mixer=cfg.mixer, task=cfg.task, train=cfg.train, model_cfg=model_cfg,
            points=cfg.points, d_model=cfg.d_model, sweep_id=sweep_id,
            seed=cfg.seed, state_size=state_size, num_parameters=num_params, peak_lr=peak_lr,
            fingerprint=fingerprint, device=device)

        run = None
        if wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity,
                name=f"{cfg.mixer}-{label}-state{state_size//1000}k", group=group, job_type="train",
                mode=wandb_mode, config=full_config, reinit=True,
                tags=["mqar", cfg.mixer, "state-size-sweep"],
            )
            run.log({"state_size": state_size, "num_parameters": num_params})
            # also persist the full config as a downloadable artifact for full reproducibility
            with open("config.json", "w") as f:
                json.dump(full_config, f, indent=2)
            run.save("config.json")

        logger = WandbLogger(run)
        metrics = train_one_run(model, train_dl, test_dl, logger, peak_lr=peak_lr,
                                train=cfg.train, device=device, use_amp=use_amp)

        if run is not None:
            run.summary.update({"state_size": state_size, "num_parameters": num_params,
                                "peak_learning_rate": peak_lr, **metrics})
            run.finish()

        results.append({
            "label": label,
            "state_size": int(state_size),
            "num_parameters": int(num_params),
            "learning_rate": peak_lr,
            "valid_accuracy": metrics["valid/accuracy"],
            "valid_best_accuracy": metrics["valid/best_accuracy"],
        })

    print("\nSweep complete.")
    for r in results:
        print(f"  state_size={r['state_size']:>11d}  {r['label']:<12s}  "
              f"lr={r['learning_rate']:.2e}  best_acc={r['valid_best_accuracy']:.4f}")

    plot_results(cfg, results, sweep_id=sweep_id, group=group, wandb_mode=wandb_mode)
    return results


def plot_results(cfg: SweepConfig, results: list[dict], *, sweep_id: str, group: str, wandb_mode: str,
                 out_path: str = "mqar_state_size_vs_accuracy.png"):
    """Accuracy (linear y) vs. state size (log x) -- the canonical zoology view."""
    import matplotlib.pyplot as plt

    label = MIXER_LABEL.get(cfg.mixer, cfg.mixer)
    color = MIXER_COLOR.get(cfg.mixer, "#3b76af")

    results_sorted = sorted(results, key=lambda r: r["state_size"])
    xs = [r["state_size"] for r in results_sorted]
    ys = [r["valid_best_accuracy"] for r in results_sorted]  # best val accuracy (robust to overfitting tail)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("State size (bytes)")
    ax.set_ylabel("MQAR recall accuracy (best)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"MQAR: State Size vs. Recall Accuracy\n({label}, seq_len={cfg.task.input_seq_len}, "
                 f"{cfg.task.num_kv_pairs} KV pairs, vocab={cfg.task.vocab_size})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for x, y, lbl in zip(xs, ys, [r["label"] for r in results_sorted]):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    if wandb_mode != "disabled":
        import wandb

        summary = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             name=f"summary-{sweep_id}", group=group, job_type="summary",
                             mode=wandb_mode, reinit=True, tags=["mqar", cfg.mixer, "summary"])
        table = wandb.Table(
            columns=["state_size", "valid_best_accuracy", "valid_accuracy", "label",
                     "learning_rate", "num_parameters"],
            data=[[r["state_size"], r["valid_best_accuracy"], r["valid_accuracy"], r["label"],
                   r["learning_rate"], r["num_parameters"]] for r in results_sorted])
        summary.log({
            "state_size_vs_accuracy_plot": wandb.Image(fig),
            "results": table,
            "state_size_vs_accuracy": wandb.plot.line(table, "state_size", "valid_best_accuracy",
                                                      title="MQAR: state size vs recall accuracy"),
        })
        summary.finish()
        print("Logged summary plot + table to Weights & Biases.")
