"""Analytical-ground-truth tests for the differential-geometry helpers.

The implicit-surface curvature formulas in :mod:`shared_geometry` produce
non-trivial expressions of gradient and Hessian rows. The standard sanity
check in graphics / geometry papers is to feed exact gradient + Hessian
data from a known analytical surface and verify the curvatures match the
closed-form values.

Surfaces tested here:

- **Sphere** ``f(x,y,z) = x^2 + y^2 + z^2 - R^2``. Gradient ``2(x,y,z)``,
  Hessian ``2 I``. With the outward-pointing gradient our formula gives
  mean curvature ``-1/R``, Gaussian curvature ``+1/R^2``, both principal
  curvatures ``-1/R``.

- **Cylinder** ``f(x,y,z) = x^2 + y^2 - R^2`` (axis along z). Gradient
  ``(2x, 2y, 0)``, Hessian ``diag(2, 2, 0)``. Mean curvature ``-1/(2R)``,
  Gaussian curvature ``0``, principal curvatures ``(0, -1/R)``.

- **Torus** ``f(x,y,z) = (sqrt(x^2+y^2) - R)^2 + z^2 - r^2`` with major
  radius ``R`` and minor radius ``r``. At the "outer equator" point
  ``(R+r, 0, 0)``, Gaussian curvature is ``1 / (r * (R+r))``, mean
  curvature is ``-(R + 2r) / (2 * r * (R+r))``.

Sign conventions follow the implementation: outward gradient
(``∇f`` pointing away from solid) yields negative mean curvature on
convex surfaces. Magnitudes are what matter for the tests.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_geometry import (
    compute_gaussian_curvature,
    compute_mean_curvature,
    compute_principal_curvatures,
)

ATOL: float = 1e-5
"""Absolute tolerance for curvature comparisons. Tight because all test
inputs are analytical — there is no floating-point degradation budget."""


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


def _sphere_jet(point: tuple[float, float, float]) -> tuple[torch.Tensor, ...]:
    """Return (grads, hx2, hy2, hz2) for the unit-sphere implicit f at point."""
    x, y, z = point
    grads = torch.tensor([[2 * x, 2 * y, 2 * z]], dtype=torch.float32)
    # Hessian of x^2+y^2+z^2-R^2 is constant 2 I; per-row representation:
    hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    hz2 = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    return grads, hx2, hy2, hz2


def test_sphere_gaussian_curvature_is_inverse_radius_squared():
    """Unit sphere should have K = 1.0; sphere of radius 3 should give 1/9."""
    for r in (1.0, 0.5, 3.0):
        # Scale a point on the sphere to lie on the surface of radius r.
        grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
        hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        hz2 = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)

        kg = compute_gaussian_curvature(hx2, hy2, hz2, grads).squeeze().item()
        assert math.isclose(kg, 1.0 / (r * r), abs_tol=ATOL), (
            f"sphere R={r}: K_gauss expected {1.0 / (r * r):.6f}, got {kg:.6f}"
        )


def test_sphere_mean_curvature_is_minus_inverse_radius():
    """Outward-gradient convention: H = -1/R on a sphere of radius R."""
    for r in (1.0, 0.5, 3.0):
        grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
        hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        hz2 = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)

        km = compute_mean_curvature(hx2, hy2, hz2, grads).squeeze().item()
        assert math.isclose(km, -1.0 / r, abs_tol=ATOL), f"sphere R={r}: H_mean expected {-1.0 / r:.6f}, got {km:.6f}"


def test_sphere_principal_curvatures_are_equal():
    """Sphere has both principal curvatures equal to mean curvature.

    The two are not bit-identical because ``compute_principal_curvatures``
    adds ``epsilons[0] = 1e-7`` under the discriminant before taking the
    square root. That spreads ``K1`` and ``K2`` by ``2 * sqrt(epsilons[0]) ≈
    6e-4`` even on a sphere where the discriminant is mathematically zero.
    The test tolerance reflects this expected, principled noise floor.
    """
    grads, hx2, hy2, hz2 = _sphere_jet((1.0, 0.0, 0.0))
    k1k2 = compute_principal_curvatures(hx2, hy2, hz2, grads).squeeze().tolist()
    assert math.isclose(k1k2[0], k1k2[1], abs_tol=1e-3), f"sphere: K1, K2 should be equal, got {k1k2}"
    # Both ~= -1 on the unit sphere with outward gradient.
    assert math.isclose(k1k2[0], -1.0, abs_tol=1e-3), f"unit sphere: K1 expected -1, got {k1k2[0]}"


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------


def test_cylinder_gaussian_curvature_is_zero():
    """A cylinder is a developable surface: K_gauss must be exactly zero."""
    for r in (1.0, 2.5, 0.7):
        # Point on the cylinder of radius r, axis aligned with z.
        grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
        # Hessian of x^2+y^2-R^2: diag(2, 2, 0).
        hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        hz2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        kg = compute_gaussian_curvature(hx2, hy2, hz2, grads).squeeze().item()
        assert math.isclose(kg, 0.0, abs_tol=ATOL), f"cylinder R={r}: K_gauss expected 0, got {kg}"


def test_cylinder_mean_curvature_is_minus_half_inverse_radius():
    """Outward gradient: cylinder mean curvature is -1/(2R)."""
    for r in (1.0, 2.5, 0.7):
        grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
        hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        hz2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        km = compute_mean_curvature(hx2, hy2, hz2, grads).squeeze().item()
        expected = -1.0 / (2 * r)
        assert math.isclose(km, expected, abs_tol=ATOL), f"cylinder R={r}: H_mean expected {expected:.6f}, got {km:.6f}"


def test_cylinder_principal_curvatures_are_zero_and_minus_inverse_radius():
    """One principal direction along the axis (zero curvature), the other
    around the circumference (-1/R)."""
    r = 1.0
    grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
    hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    hz2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    k1k2 = compute_principal_curvatures(hx2, hy2, hz2, grads).squeeze().tolist()
    k_sorted = sorted(k1k2)  # Ascending so we know which is which.
    # Expected (-1/R, 0) = (-1, 0) on the unit cylinder.
    assert math.isclose(k_sorted[0], -1.0 / r, abs_tol=1e-3), f"cylinder K_min expected {-1.0 / r}, got {k_sorted[0]}"
    assert math.isclose(k_sorted[1], 0.0, abs_tol=1e-3), f"cylinder K_max expected 0, got {k_sorted[1]}"


# ---------------------------------------------------------------------------
# Torus (outer equator)
# ---------------------------------------------------------------------------


def test_torus_outer_equator_matches_known_curvatures():
    """At the outer equator of a torus, K_gauss = 1/(r*(R+r)), K_mean has a
    known closed form. This stresses the formulas with non-axis-aligned
    Hessian behaviour."""
    # Torus: f(x,y,z) = (sqrt(x^2+y^2) - R)^2 + z^2 - r^2
    # At point (R+r, 0, 0):
    #   d/dx = 2 * (sqrt(x^2+y^2) - R) * (x / sqrt(x^2+y^2)) = 2r * 1 = 2r
    #   d/dy = 0, d/dz = 0
    #
    # Hessian at (R+r, 0, 0):
    #   fxx = 2  (d^2/dx^2 at the outer equator simplifies to 2)
    #   fyy = 2 * r / (R + r)
    #   fzz = 2
    #   off-diagonals = 0
    R, r_minor = 2.0, 0.5
    fx = 2 * r_minor  # = 2 * (sqrt(x^2) - R) * (x/sqrt(x^2)) at y=z=0
    # The second-derivative-by-hand for the squared-distance-to-axis form is
    # d2/dx2 = 2 at the outer equator; d2/dy2 = 2 * r_minor / (R + r_minor);
    # d2/dz2 = 2.
    fxx = 2.0
    fyy = 2.0 * r_minor / (R + r_minor)
    fzz = 2.0

    grads = torch.tensor([[fx, 0.0, 0.0]], dtype=torch.float32)
    hx2 = torch.tensor([[fxx, 0.0, 0.0]], dtype=torch.float32)
    hy2 = torch.tensor([[0.0, fyy, 0.0]], dtype=torch.float32)
    hz2 = torch.tensor([[0.0, 0.0, fzz]], dtype=torch.float32)

    kg = compute_gaussian_curvature(hx2, hy2, hz2, grads).squeeze().item()
    expected_kg = 1.0 / (r_minor * (R + r_minor))
    assert math.isclose(kg, expected_kg, abs_tol=1e-4), (
        f"torus outer equator (R={R}, r={r_minor}): K_gauss expected {expected_kg:.6f}, got {kg:.6f}"
    )

    # The principal curvatures at the outer equator are
    #   k1 = -1/r  (around the minor circle, with outward gradient)
    #   k2 = -1/(R+r)  (around the major circle)
    k1k2 = compute_principal_curvatures(hx2, hy2, hz2, grads).squeeze().tolist()
    k_sorted = sorted(k1k2)
    expected_k_min = -1.0 / r_minor
    expected_k_max = -1.0 / (R + r_minor)
    assert math.isclose(k_sorted[0], expected_k_min, abs_tol=1e-3), (
        f"torus K_min expected {expected_k_min:.4f}, got {k_sorted[0]:.4f}"
    )
    assert math.isclose(k_sorted[1], expected_k_max, abs_tol=1e-3), (
        f"torus K_max expected {expected_k_max:.4f}, got {k_sorted[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Numerical stability across mesh scales
# ---------------------------------------------------------------------------


def test_curvature_is_scale_invariant_to_position():
    """The sphere's K should not depend on which point of the sphere you
    sample. This catches subtle bugs where the formula uses point coords
    where it should use only Hessian + gradient."""
    points_on_unit_sphere = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0 / math.sqrt(3),) * 3,
        (-0.5, math.sqrt(0.75), 0.0),
    ]
    expected_k = 1.0  # Gaussian curvature of unit sphere.

    for point in points_on_unit_sphere:
        grads, hx2, hy2, hz2 = _sphere_jet(point)
        kg = compute_gaussian_curvature(hx2, hy2, hz2, grads).squeeze().item()
        assert math.isclose(kg, expected_k, abs_tol=ATOL), f"sphere at {point}: K_gauss expected {expected_k}, got {kg}"


def test_curvature_handles_small_mesh_scale():
    """Sphere of radius 0.01 (mm-scale geometry under mesh normalisation).
    Currently the only goal is to verify the DENOM_FLOOR guard doesn't
    silently swallow the answer."""
    r = 0.01
    grads = torch.tensor([[2 * r, 0.0, 0.0]], dtype=torch.float32)
    hx2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    hy2 = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    hz2 = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)

    kg = compute_gaussian_curvature(hx2, hy2, hz2, grads).squeeze().item()
    expected_kg = 1.0 / (r * r)  # 10_000
    relative_error = abs(kg - expected_kg) / expected_kg
    assert relative_error < 1e-3, (
        f"small-scale sphere R={r}: K_gauss relative error {relative_error:.4e} (got {kg}, expected {expected_kg})"
    )
