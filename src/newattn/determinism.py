"""Global determinism + device helpers (mirrors zoology.utils.set_determinism)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_determinism(seed: int):
    """Make a run reproducible end-to-end (mirrors zoology.utils.set_determinism + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
