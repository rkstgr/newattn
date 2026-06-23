"""Experiment configuration dataclasses.

These mirror the config blocks from the original notebooks, unified so a single
`SweepConfig` drives any mixer. All knobs are plain dataclass fields so they are
easy to edit in a script or override from the command line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Default state-size sweep (x-axis) per mixer. Each width maps to one point on the
# state-size axis; see `newattn.mixers` for the per-mixer state-size formula.
DEFAULT_D_MODELS: dict[str, list[int]] = {
    "attention": [8, 16, 32, 48, 64, 128],
    "mamba2": [8, 16, 32, 48, 64],
    # `gdn2` (pure-PyTorch) is single-head with head_dim = d_model, so state ~ d_model**2
    # (8 * d_model**2 bytes at n_layers=2, expand_v=1); any width works. `gdn2_triton` keeps
    # a fixed head_dim=64 (num_heads = d_model // 64), so its widths must be multiples of 64.
    "gdn2": [64, 128, 192, 256, 320],
    "gdn2_triton": [64, 128, 192, 256, 320],
}

# Peak learning rate per d_model, hand-tuned per mixer (one entry per width above).
# Override on the command line with `--lr` (flat LR for the whole sweep) or by editing
# the `lr_per_d_model` on each experiment's SweepConfig.
DEFAULT_LR_PER_D_MODEL: dict[str, dict[int, float]] = {
    "attention": {8: 1.5e-3, 16: 1.5e-3, 32: 1e-3, 48: 1e-3, 64: 7.66e-4, 128: 3.83e-4, 192: 2.55e-4},
    "mamba2": {8: 3e-3, 16: 2e-3, 32: 1e-3, 48: 1e-3, 64: 8e-4},
    "gdn2": {32: 1e-3, 48: 1e-4, 64: 8e-4, 128: 5e-4},
    "gdn2_triton": {64: 7.66e-4, 128: 5e-4, 192: 5e-4, 256: 5e-4, 320: 5e-4},
}


@dataclass
class MQARTaskConfig:
    """Fixed MQAR difficulty (mirrors a canonical zoology train/test segment)."""

    vocab_size: int = 8192
    input_seq_len: int = 128
    num_kv_pairs: int = 8
    power_a: float = 0.01
    random_non_queries: bool = True
    num_train_examples: int = 50_000
    num_test_examples: int = 1_000


@dataclass
class ModelConfig:
    """Decoder-only sequence model with a selectable sub-quadratic sequence mixer.

    Mirrors `zoology.model.LanguageModel`'s config. It carries the union of every
    mixer's hyper-parameters; the unused ones are simply ignored by the active
    mixer (and logged for a complete, reproducible record).
    """

    d_model: int = 128
    n_layers: int = 2
    mixer: str = "attention"  # "attention" | "mamba2" | "gdn2" (pure-PyTorch) | "gdn2_triton"

    # ---- Attention (MHA) sequence-mixer hyper-parameters ----
    num_heads: int = 1
    attn_dropout: float = 0.0

    # ---- Mamba2 sequence-mixer hyper-parameters ----
    d_state: int = 128  # SSM state dimension N (the recurrent-memory knob)
    d_conv: int = 4  # depthwise causal conv width
    expand: int = 2  # inner expansion: d_inner = expand * d_model
    headdim: int = 64  # target head dim (auto-shrunk to divide d_inner)
    ngroups: int = 1  # B/C groups (shared across heads when ngroups=1)

    # ---- Gated DeltaNet 2 sequence-mixer hyper-parameters (fla.layers.GatedDeltaNet2) ----
    gdn2_head_dim: int = 64  # per-head K dim (also V dim when expand_v=1); the memory knob
    gdn2_num_heads: int | None = None  # default: d_model // gdn2_head_dim
    gdn2_expand_v: float = 1.0  # value expansion: head_v_dim = head_dim * expand_v
    gdn2_mode: str = "chunk"  # "chunk" (training) | "fused_recurrent"
    gdn2_use_short_conv: bool = True  # short causal conv on q/k/v (as in fla)
    gdn2_conv_size: int = 4
    gdn2_allow_neg_eigval: bool = False

    # ---- shared ----
    mlp_hidden_mult: int = 4
    vocab_size: int = 8192
    max_position_embeddings: int = 128
    resid_dropout: float = 0.0
    embed_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    learnable_word_embeddings: bool = True
    block_type: str = "TransformerBlock"
    sequence_mixer: str = ""  # set in __post_init__ from `mixer`
    state_mixer: str = "zoology.mixers.mlp.MLP"

    def __post_init__(self):
        self.sequence_mixer = {
            "attention": "zoology.mixers.attention.MHA",
            "mamba2": "zoology.mixers.mamba2.Mamba2",
            "gdn2": "newattn.mixers.gdn2.GatedDeltaNet2Naive",
            "gdn2_triton": "fla.layers.gdn2.GatedDeltaNet2",
        }[self.mixer]
        if self.gdn2_num_heads is None:
            self.gdn2_num_heads = max(1, self.d_model // self.gdn2_head_dim)


@dataclass
class TrainParams:
    """Training hyper-parameters (zoology/train.py + warmup, grad-clip, patience).

    The peak learning rate is *not* fixed here -- it is set per run from the
    experiment's `lr_per_d_model` map. The schedule is a linear warmup over
    `warmup_epochs` followed by per-step cosine decay to 0.
    """

    max_epochs: int = 32
    weight_decay: float = 0.1
    batch_size: int = 256
    test_batch_size: int = 256
    warmup_epochs: float = 1.0  # linear LR warmup over this many epochs
    grad_clip: float = 1.0  # max gradient norm (<= 0 disables clipping)
    amp_dtype: str = "bfloat16"  # autocast dtype when use_amp: "bf16"/"fp16"/"fp32" (fp16 for Turing/T4)
    early_stopping_metric: str = "valid/accuracy"
    early_stopping_threshold: float = 0.99  # stop early once the task is solved
    patience: int = 5  # stop if valid/accuracy hasn't improved for N epochs
    seed: int = 123


@dataclass
class SweepConfig:
    """A full state-size sweep: one training run per width in `d_models`."""

    mixer: str = "attention"  # "attention" | "mamba2" | "gdn2" (pure-PyTorch) | "gdn2_triton"
    exp_id: str = "exp"  # short tag used in W&B group / run names
    d_models: list[int] = field(default_factory=lambda: list(DEFAULT_D_MODELS["attention"]))
    # Peak learning rate per d_model (one entry for every width in `d_models`).
    lr_per_d_model: dict[int, float] = field(default_factory=dict)
    seed: int = 123

    task: MQARTaskConfig = field(default_factory=MQARTaskConfig)
    train: TrainParams = field(default_factory=TrainParams)

    # ---- Weights & Biases ----
    wandb_project: str = "zoology-mqar"
    wandb_entity: str | None = None  # your W&B entity (username/team), or None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    group: str | None = None  # W&B group; defaults to f"mqar-{mixer}-{exp_id}"

    # Optional per-model-config overrides applied to every ModelConfig in the sweep
    # (e.g. {"d_state": 64} to sweep at a different Mamba2 state dim).
    model_overrides: dict = field(default_factory=dict)

    def lr_for(self, d_model: int) -> float:
        try:
            return self.lr_per_d_model[d_model]
        except KeyError:
            raise KeyError(
                f"no learning rate for d_model={d_model}; add it to lr_per_d_model "
                f"(have {sorted(self.lr_per_d_model)}) or pass --lr to set a flat LR"
            ) from None

    def resolved_group(self) -> str:
        return self.group or f"mqar-{self.mixer}-{self.exp_id}"
