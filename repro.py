"""Reproducibility helpers: device resolution and global RNG seeding.

`resolve_device` picks an explicit torch device from a config string, falling
back to the best available accelerator when given ``"auto"`` or an empty
string. `set_global_seed` seeds every RNG used by the training pipelines.

These are intentionally kept in one small module so a future pixi-based or
mypy-strict pass does not have to chase device handling across the codebase.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def default_device() -> torch.device:
    """Return the best available accelerator: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str | None) -> torch.device:
    """Resolve a config device string.

    ``"auto"`` or an empty value picks the best available accelerator.
    Any other value is passed through to ``torch.device``; if the user
    explicitly asked for ``"cuda"`` on a machine without CUDA we fail
    loudly rather than silently falling back, because silently swapping
    the device alters wall-clock cost and is something the user should
    know about.
    """
    if not name or name == "auto":
        return default_device()

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests device='cuda' but torch reports CUDA unavailable. "
            "Set 'device' to 'auto', 'mps', or 'cpu' in the config."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Config requests device='mps' but torch reports MPS unavailable. "
            "Set 'device' to 'auto', 'cuda', or 'cpu' in the config."
        )
    return device


def set_global_seed(seed: int) -> None:
    """Seed every RNG used by the training pipelines.

    Bit-exact same-machine reproducibility on CUDA additionally requires
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the environment because cuBLAS
    uses non-deterministic atomic-add reductions by default. MPS and CPU
    are deterministic once the RNGs are seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
