"""Smoke tests for the cross-device reproducibility layer.

These tests do not retrain anything — they validate that:

- ``repro.resolve_device`` picks a real torch device and refuses impossible ones.
- ``repro.set_global_seed`` produces bit-exact RNG sequences across runs.
- A SIREN layer field with the second-order autograd path used by
  ``shared_geometry.computePrincipalCurvatures`` can run on the device that
  ``resolve_device("auto")`` selects on the host. This is the path the
  ``support_free_pipeline`` / ``toolpath_alignment_pipeline`` rely on.
- The shipped fertility checkpoint loads cleanly with ``map_location`` set
  to the resolved device. This is the Tier 0 visualization path.

Run with: ``pytest tests/test_smoke.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro import default_device, resolve_device, set_global_seed
from siren_pytorch import SirenNet

SHIPPED_FERTILITY_CHECKPOINT = ROOT / "examples" / "checkpoints" / "parametersTest_batched_fertility_10_128_7_7.pt"


def test_resolve_device_auto_picks_real_device():
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}


def test_resolve_device_empty_string_is_auto():
    assert resolve_device("").type in {"cuda", "mps", "cpu"}


def test_resolve_device_explicit_cpu_always_works():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot test the negative path")
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        resolve_device("cuda")


def test_resolve_device_rejects_unavailable_mps():
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available; cannot test the negative path")
    with pytest.raises(RuntimeError, match="MPS unavailable"):
        resolve_device("mps")


def test_set_global_seed_is_deterministic():
    set_global_seed(123)
    a_torch = torch.rand(8)
    a_numpy = np.random.rand(8)

    set_global_seed(123)
    b_torch = torch.rand(8)
    b_numpy = np.random.rand(8)

    assert torch.equal(a_torch, b_torch), "torch.rand differs after re-seeding"
    assert np.array_equal(a_numpy, b_numpy), "np.random.rand differs after re-seeding"


def test_siren_second_order_autograd_on_resolved_device():
    """The critical SIREN+curvature path: forward + first + second order grads."""
    device = default_device()
    set_global_seed(0)

    net = SirenNet(3, 32, 1, 4, w0=7.0, w0_initial=7.0).to(device)
    x = torch.randn(16, 3, device=device, requires_grad=True)

    out = net(x)
    assert out["scalars"].shape == (16, 1)
    assert out["grads"].shape == (16, 3)
    assert out["HX2"].shape == (16, 3)
    assert out["HY2"].shape == (16, 3)
    assert out["HZ2"].shape == (16, 3)
    assert out["scalars"].device.type == device.type


def test_shipped_fertility_checkpoint_loads_on_resolved_device():
    if not SHIPPED_FERTILITY_CHECKPOINT.exists():
        pytest.skip(f"shipped checkpoint not present at {SHIPPED_FERTILITY_CHECKPOINT}")

    device = default_device()
    net = SirenNet(3, 128, 1, 10, w0=7.0, w0_initial=7.0).to(device)
    # The shipped checkpoint is a state_dict produced by torch.save(model.state_dict()).
    state = torch.load(SHIPPED_FERTILITY_CHECKPOINT, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state)

    x = torch.randn(8, 3, device=device, requires_grad=True)
    out = net(x)
    assert torch.isfinite(out["scalars"]).all()
    assert torch.isfinite(out["grads"]).all()
    assert torch.isfinite(out["HX2"]).all()
