"""Reproducibility helpers."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy, PYTHONHASHSEED and (if installed) torch/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when available, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
