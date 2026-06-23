"""Run a state-size-vs-recall sweep and plot the recreated zoology curve.

`run_sweep(SweepConfig)` trains one model per width in `cfg.d_models`, maps each width to
a point on the state-size x-axis, logs everything to W&B, and produces the final
state-size-vs-accuracy figure (saved to disk and logged as a W&B image / table / line).
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

MIXER_LABEL = {"attention": "Transformer (attention)", "mamba2": "Mamba2", "gdn2": "Gated DeltaNet 2"}
MIXER_COLOR = {"attention": "#3b76af", "mamba2": "#c4694b", "gdn2": "#4b78c4"}


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

    # Planning table: each width -> a state size (the x-axis) and a width-scaled peak LR.
    n_layers = ModelConfig().n_layers
    print(f"Planned sweep (mixer={cfg.mixer!r}; state size is the x-axis):")
    for d in cfg.d_models:
        mc = ModelConfig(d_model=d, mixer=cfg.mixer, **cfg.model_overrides)
        state = spec.state_size_bytes(mc, n_layers, cfg.task.input_seq_len)
        print(f"  d_model={d:>4d}  {spec.dims_str(mc)}  ->  state_size={state:>11d}  "
              f"->  lr={cfg.lr_for(d):.2e}")

    wandb_mode = maybe_login(cfg.wandb_mode)
    use_amp = spec.use_amp and device == "cuda"

    results = []
    for i, d_model in enumerate(cfg.d_models):
        peak_lr = cfg.lr_for(d_model)
        print(f"\n=== Run {i + 1}/{len(cfg.d_models)}: d_model={d_model} (lr={peak_lr:.2e}) ===")
        set_determinism(cfg.seed)

        # data (identical task across runs) + model
        train_dl, test_dl, fingerprint = build_dataloaders(
            cfg.task, seed=cfg.seed, batch_size=cfg.train.batch_size,
            test_batch_size=cfg.train.test_batch_size)
        model_cfg = ModelConfig(d_model=d_model, mixer=cfg.mixer, vocab_size=cfg.task.vocab_size,
                                max_position_embeddings=cfg.task.input_seq_len, **cfg.model_overrides)
        model = LanguageModel(model_cfg)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_size = model.state_size(sequence_length=cfg.task.input_seq_len)
        full_config = build_full_config(
            mixer=cfg.mixer, task=cfg.task, train=cfg.train, model_cfg=model_cfg,
            lr_per_d_model=cfg.lr_per_d_model, d_models=cfg.d_models, sweep_id=sweep_id,
            seed=cfg.seed, state_size=state_size, num_parameters=num_params, peak_lr=peak_lr,
            fingerprint=fingerprint, device=device)

        run = None
        if wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity,
                name=f"d_model{d_model}-state{state_size}", group=group, job_type="train",
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
            "d_model": d_model,
            "state_size": int(state_size),
            "num_parameters": int(num_params),
            "learning_rate": peak_lr,
            "valid_accuracy": metrics["valid/accuracy"],
            "valid_best_accuracy": metrics["valid/best_accuracy"],
        })

    print("\nSweep complete.")
    for r in results:
        print(f"  state_size={r['state_size']:>11d}  d_model={r['d_model']:>4d}  "
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
    for x, y, d in zip(xs, ys, [r["d_model"] for r in results_sorted]):
        ax.annotate(f"d={d}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    if wandb_mode != "disabled":
        import wandb

        summary = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             name=f"summary-{sweep_id}", group=group, job_type="summary",
                             mode=wandb_mode, reinit=True, tags=["mqar", cfg.mixer, "summary"])
        table = wandb.Table(
            columns=["state_size", "valid_best_accuracy", "valid_accuracy", "d_model",
                     "learning_rate", "num_parameters"],
            data=[[r["state_size"], r["valid_best_accuracy"], r["valid_accuracy"], r["d_model"],
                   r["learning_rate"], r["num_parameters"]] for r in results_sorted])
        summary.log({
            "state_size_vs_accuracy_plot": wandb.Image(fig),
            "results": table,
            "state_size_vs_accuracy": wandb.plot.line(table, "state_size", "valid_best_accuracy",
                                                      title="MQAR: state size vs recall accuracy"),
        })
        summary.finish()
        print("Logged summary plot + table to Weights & Biases.")
