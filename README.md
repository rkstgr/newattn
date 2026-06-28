# newattn

A minimal, modular recreation of the central experiment from
**[Zoology](https://github.com/HazyResearch/zoology)** (Arora, Eyuboglu, et al.,
*"Zoology: Measuring and Improving Recall in Efficient Language Models"*): the
**state-size vs. recall-accuracy** curve on the **Multi-Query Associative Recall (MQAR)**
task, for several sequence mixers.

Each experiment trains one decoder-only model per width, maps each width to a point on the
**state-size x-axis**, and plots recall accuracy against state size (log x). The three
notebooks (`mqar_exp00*.ipynb`) have been refactored into one reusable package plus thin
experiment scripts.

| Experiment | Mixer | State size (bytes) | Notes |
|---|---|---|---|
| `exp001b` | Transformer (attention) | `4·n_layers·2·d_model·seq_len` | unbounded state (KV cache) |
| `exp002` | Mamba2 (pure PyTorch) | `4·n_layers·expand·d_model·d_state` | bounded state |
| `exp003` | Gated DeltaNet 2 (fla) | `4·n_layers·num_heads·head_dim·head_v_dim` | bounded state |
| `exp004` | Titans (neural memory) | `4·n_layers·num_heads·2·head_dim·mem_hidden` | bounded state (MLP fast weights) |

## Layout

```
src/newattn/
  config.py      # MQARTaskConfig, ModelConfig, TrainParams, SweepConfig
  data.py        # MQAR generator + dataloaders (ported from zoology)
  model.py       # decoder-only LanguageModel harness (zoology.model.LanguageModel)
  mixers/        # pluggable sequence mixers behind a small registry
    attention.py #   MHA (zoology.mixers.attention)
    mamba2.py    #   pure-PyTorch Mamba2 SSD scan
    gdn2.py      #   wrapper over fla.layers.GatedDeltaNet2
    titans.py    #   pure-PyTorch Titans MLP neural memory
  train.py       # training loop (zoology/train.py + warmup/clip/early-stop)
  tracking.py    # Weights & Biases harness
  sweep.py       # run_sweep() + the recreated plot
  cli.py         # shared CLI / env-var overrides for the experiment scripts
experiments/
  exp001b_transformer.py
  exp002_mamba2.py
  exp003_gdn2.py
  exp004_titans.py
```

## Configure

Every knob lives in a dataclass (`src/newattn/config.py`); each experiment script sets its
defaults in a `SweepConfig` at the top of the file:

- **`MQARTaskConfig`** — task difficulty: `vocab_size`, `input_seq_len`, `num_kv_pairs`, …
- **`ModelConfig`** — `d_model`, `n_layers`, and mixer-specific knobs (`d_state`/`expand`
  for Mamba2, `gdn2_head_dim`/`gdn2_expand_v` for GDN2, `num_heads` for attention).
- **`TrainParams`** — `max_epochs`, `batch_size`, `weight_decay`, `grad_clip`, early stop.
- **`SweepConfig`** — ties it together: `mixer`, `d_model` (the fixed residual-stream width),
  `points` (the list of `SweepPoint`s that sweep the state size), `seed`, and W&B
  `wandb_project` / `wandb_entity` / `wandb_mode`.
- **`SweepPoint`** — one run: model-config `overrides` (the state knobs to set at the fixed
  `d_model`, e.g. `{"d_state": 16}` or `{"gdn2_head_dim": 32, "gdn2_expand_v": 2}`), its peak
  `lr`, and an optional `label`. State size is derived from the resulting `ModelConfig`, so it
  is **decoupled from `d_model`** — the x-axis is the recurrent state, not the model width.

Per-mixer default sweep points live in `DEFAULT_POINTS` in `config.py`.

Edit those defaults directly, **or** override common knobs without touching files via CLI
flags / environment variables (handy in Colab). Run any script with `--help`:

```bash
python experiments/exp002_mamba2.py --help
python experiments/exp002_mamba2.py --wandb-mode disabled            # default points
python experiments/exp002_mamba2.py --d-model 48 --lr 5e-4           # fix a different width + flat LR
python experiments/exp003_gdn2.py   --mixer mamba2                   # swap mixer, keep the harness
```

`--d-model` sets the fixed residual-stream width for the whole sweep; `--lr` overrides every
point's peak LR. To change which state configurations are swept, edit the `points` list (or
`DEFAULT_POINTS[mixer]`).

Recognised env vars: `WANDB_MODE` (`online`/`offline`/`disabled`), `WANDB_ENTITY`,
`WANDB_PROJECT` (CLI flags win over env vars, which win over the script defaults).

## Run locally with pixi

[pixi](https://pixi.sh) manages the environment. Python comes from conda; everything else
is pip wheels, so the local env matches Colab's.

```bash
pixi install                      # create the default environment
pixi run exp001b                  # transformer sweep
pixi run exp002                   # mamba2 sweep
# pass extra flags through the task:
pixi run exp001b --wandb-mode disabled
```

`exp003` (Gated DeltaNet 2) needs `flash-linear-attention` + a CUDA GPU, so it lives in a
separate `gdn2` environment:

```bash
pixi run -e gdn2 python experiments/exp003_gdn2.py        # requires an NVIDIA GPU
```

> The default pixi/conda PyTorch is a **CPU** build — fine for `exp001b` and `exp002`.
> For GPU training (and for `gdn2`), use Colab or a CUDA box; see below.

## Run in Google Colab

`gdn2` needs a GPU, and Colab is the easiest way to get one. Attention and Mamba2 also run
much faster there.

1. **Set the runtime to GPU:** *Runtime → Change runtime type → T4 GPU* (required for
   `exp003`; optional but recommended for the others).

2. **Clone + install** (first cell):

   ```python
   !git clone https://github.com/rkstgr/newattn.git
   %cd newattn
   !pip install -e ".[gdn2]"      # drop the [gdn2] extra if you only run exp001b/exp002
   ```

   Colab already ships a CUDA build of PyTorch; `pip install -e .` reuses it and only adds
   the missing packages (`wandb`, `einops`, and — with `[gdn2]` — `flash-linear-attention`
   + Triton).

3. **Run an experiment** (second cell). Weights & Biases will prompt you to log in the
   first time; set your entity, or disable W&B entirely:

   ```python
   # Transformer (attention)
   !python experiments/exp001b_transformer.py --wandb-entity YOUR_WANDB_ENTITY

   # Mamba2
   !python experiments/exp002_mamba2.py --wandb-entity YOUR_WANDB_ENTITY

   # Gated DeltaNet 2 (needs the GPU runtime)
   !python experiments/exp003_gdn2.py --wandb-entity YOUR_WANDB_ENTITY

   # …or run without Weights & Biases:
   !python experiments/exp002_mamba2.py --wandb-mode disabled
   ```

   Each run prints the planned sweep, trains one model per width, writes
   `mqar_state_size_vs_accuracy.png`, and (unless disabled) logs the curve + full config to
   W&B. Quick smoke test: add `--max-epochs 1 --num-train-examples 2000`.

   To configure inside a notebook cell instead of editing the script, set env vars first:

   ```python
   import os
   os.environ["WANDB_MODE"] = "disabled"      # or "online"
   os.environ["WANDB_ENTITY"] = "YOUR_ENTITY"
   !python experiments/exp001b_transformer.py
   ```

## Adding a mixer

The mixers are a registry (`src/newattn/mixers/`). To add one:

1. Write an `nn.Module` with `forward(x: (b,l,d)) -> (b,l,d)` and a
   `state_size(seq_len) -> int` method (element count for one layer).
2. Provide `build(cfg, layer_idx)`, `state_size_bytes(cfg, n_layers, seq_len)`, and
   `dims_str(cfg)`, plus a module-level `SPEC = MixerSpec(...)`.
3. Register the `SPEC` in `mixers/__init__.py` and add a default state-size sweep
   (a list of `SweepPoint`s varying the mixer's state knobs) to `DEFAULT_POINTS` in `config.py`.

It then drops straight into the shared model/train/sweep harness.

## Faithfulness to zoology

The MQAR generator is a near-verbatim port of `zoology/data/multiquery_ar.py`; the model
mirrors `zoology/model.py` (`block_type="TransformerBlock"`, `state_mixer=…mlp.MLP`); the
training loop mirrors `zoology/train.py`; and `LanguageModel.state_size` follows the
zoology convention (sum per-layer state × 4 bytes). The mixers are reimplemented (rather
than `pip install zoology`) because the package hard-requires fragile CUDA build deps;
Mamba2 in particular is an exact O(L²) pure-PyTorch SSD scan, mathematically identical to
the chunked kernel for these sequence lengths.

On top of the zoology defaults the training loop adds: a per-step warmup→cosine LR
schedule whose peak LR is set per run from each `SweepPoint.lr`, gradient clipping, early
stopping (solved or plateau), and no weight decay on 1-D parameters.
