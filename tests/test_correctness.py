"""Correctness regression tests for three bugs in the original code.

These tests are written *before* the fixes land — they fail on the buggy
implementation, pass after the fix, and stay green as long as nobody
reintroduces the bug.

1. ``shared_geometry.supportLoss`` used the literal ``3.1457`` in place of
   ``pi``. The discrepancy (4.1e-4 absolute, ~0.13 % relative) shifts the
   support-angle threshold by ~0.17°. Same bug in ``platform_losses.py`` and
   ``collisionLoss.py``.

2. ``collisionLoss.get_cone_sample_direction_cosines3`` had a dead
   stratifier: ``cos(theta) + (1 - 1) * torch.rand(...) ** n`` collapses to
   ``cos(theta)`` for every "biased toward boundary" sample, leaving the
   cone sampler with only ~3 distinct latitudes.

3. ``field_losses.add_collision_losses`` silently skipped non-finite
   collision losses while still adding the NaN to the *logged* record,
   producing monotonically-NaN log files with no warning to the user.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collisionLoss import get_cone_sample_direction_cosines3
from field_losses import add_collision_losses
from shared_geometry import supportLoss


def test_support_loss_uses_pi_not_3p1457():
    """At dotProd just inside the corrected boundary, the loss must be > 0.

    With the buggy constant ``3.1457``, the boundary lands at
    cos(132° * 3.1457 / 180) = -0.67035. With ``math.pi`` it lands at
    cos(132° * pi / 180) = -0.66913. A dot product chosen in the gap
    (-0.66975) is *outside* the buggy boundary (relu kills the term) but
    *inside* the corrected boundary (relu lets a small positive term
    through).
    """
    target_cos = -0.66975
    normals = torch.tensor([[1.0, 0.0, 0.0]])
    grads = torch.tensor([[target_cos, math.sqrt(1.0 - target_cos**2), 0.0]])

    out = supportLoss(normals, grads, angle_degrees=132.0)
    assert out["loss"].item() > 0.0, (
        "supportLoss returned zero at a dot product that should be inside the "
        "support-angle violation region — most likely the threshold is still "
        "computed with the literal 3.1457 instead of math.pi."
    )


def test_no_3p1457_literal_in_source():
    """Regression guard: nobody should reintroduce the buggy pi substitute."""
    offenders = []
    for path in ROOT.glob("*.py"):
        if "3.1457" in path.read_text():
            offenders.append(path.name)
    assert not offenders, (
        f"Files {offenders} contain the literal 3.1457. Use math.pi, np.pi, "
        f"or torch.pi instead. The literal is wrong from the 5th significant "
        f"digit and shifts angle thresholds systematically."
    )


def test_cone_sampler_is_non_degenerate():
    """The cone sampler must produce a non-trivial spread of latitudes.

    Sampled directions have z-component equal to ``cos(alpha)`` (the cone
    is aligned with +Z). On the buggy code, the third group of samples
    collapses to ``cos(theta)`` for every point, so a 20-sample cone has
    at most 3 distinct z-values. After the fix the third group has
    randomized cosines, giving roughly one distinct value per sample.
    """
    samples = get_cone_sample_direction_cosines3(angle=60.0, m=20, device="cpu")
    z_values = samples[:, 2]
    distinct = torch.unique(z_values).numel()
    assert distinct >= 5, (
        f"Cone sampler returned only {distinct} distinct latitudes out of 20 "
        f"samples — the (1-1) dead stratifier in get_cone_sample_direction_cosines3 "
        f"is collapsing the third sample group to cos(theta)."
    )


def _make_fake_collision_returning_nan():
    """Return an object exposing the methods add_collision_losses calls."""
    nan = torch.tensor(float("nan"))

    class FakeCollision:
        def collision_scalar_loss(self, *args, **kwargs):
            return {"loss": nan}

        def collision_scalar_loss_far(self, *args, **kwargs):
            return {"loss": nan}

        def collision_scalar_loss_far2(self, *args, **kwargs):
            return {"loss": nan}

        def collision_scalar_loss_far_in(self, *args, **kwargs):
            return {"loss": nan}

    return FakeCollision()


def test_collision_nan_is_logged_not_silent(capsys):
    """A non-finite collision loss must produce a visible warning.

    On the buggy code, ``add_collision_losses`` had four
    ``if not torch.isnan(col_loss):`` guards that silently dropped the term
    from the loss while still adding the NaN to the logged record. After
    the fix, a non-finite term must emit a warning so the user knows their
    collision constraint is being skipped.
    """
    fake_collision = _make_fake_collision_returning_nan()
    data = SimpleNamespace(x_lim=1.0, y_lim=1.0, z_lim=1.0)
    config = SimpleNamespace(enable_losses=True, use_collision_loss=True)

    loss_in = torch.tensor(0.0)
    inputs = torch.zeros((4, 3))
    grads = torch.zeros((4, 3))
    out = {"scalars": torch.zeros((4, 1))}

    loss_out, record = add_collision_losses(
        loss_in, inputs, out, grads, scalar_field=None,
        collision_loss=fake_collision, data=data, config=config, epoch=10,
    )

    captured = capsys.readouterr().out.lower()
    assert "non-finite" in captured or "nan" in captured, (
        "add_collision_losses swallowed a NaN collision loss without logging "
        f"anything. captured stdout was: {captured!r}"
    )
    assert torch.isfinite(torch.as_tensor(record)).all(), (
        "When the collision term is NaN the logged record must be finite "
        f"(zeroed), not NaN — got {record!r}."
    )
