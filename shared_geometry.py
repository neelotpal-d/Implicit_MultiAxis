"""Differential-geometry helpers used by the layer / toolpath losses.

PEP 8 snake_case names; the original public API was camelCase. Functions
operate on PyTorch tensors and return PyTorch tensors so they compose
into autograd graphs.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from constants import DENOM_FLOOR

# Default principal-curvature discriminant fall-back levels. The
# discriminant ``Km**2 - Kg`` is theoretically non-negative but can dip
# below zero under floating-point error; we add a small epsilon, and if
# that still goes negative we widen by ~30x and try again, up to 1e-5.
# Each fallback emits a warning so silent precision drift surfaces.
_DEFAULT_CURVATURE_EPSILONS: tuple[float, float, float] = (1e-7, 2e-6, 1e-5)


def support_loss(
    surface_normals: torch.Tensor,
    surface_grads: torch.Tensor,
    angle_degrees: float = 132.0,
    sharpness: float = 25.0,
) -> dict:
    """Penalise boundary gradients that violate the support-angle threshold.

    Args:
        surface_normals: ``(n, 3)`` outward normals at boundary points.
        surface_grads: ``(n, 3)`` field gradients at the same points.
        angle_degrees: Maximum allowed angle between the build direction
            and the negative surface normal. Default 132° matches the
            published experiments.
        sharpness: Multiplier applied inside the hinge, before the square.

    Returns:
        ``{"loss": scalar, "mask": (n,) bool tensor of violation points}``.
    """
    grad_norm = torch.norm(surface_grads, dim=1).unsqueeze(1)
    surface_grads = surface_grads / (grad_norm + DENOM_FLOOR)
    dot_prod = torch.sum(surface_normals * surface_grads, dim=1)

    support_error = -dot_prod + np.cos(np.deg2rad(angle_degrees))
    support_error = torch.relu(sharpness * support_error)
    support_mask = support_error > 0.0
    loss = torch.mean(support_error * support_error)
    return {"loss": loss, "mask": support_mask}


def compute_gaussian_curvature(
    dx2: torch.Tensor, dy2: torch.Tensor, dz2: torch.Tensor, grads: torch.Tensor
) -> torch.Tensor:
    """Gaussian curvature of the implicit surface from Hessian rows and gradients."""
    fxx = dx2[:, 0]
    fxy = dx2[:, 1]
    fxz = dx2[:, 2]
    fyy = dy2[:, 1]
    fyz = dy2[:, 2]
    fzz = dz2[:, 2]

    h11 = fyy * fzz - fyz * fyz
    h12 = fyz * fxz - fxy * fzz
    h13 = fxy * fyz - fyy * fxz
    h22 = fxx * fzz - fxz * fxz
    h23 = fxy * fxz - fxx * fyz
    h33 = fxx * fyy - fxy * fxy

    fx = grads[:, 0]
    fy = grads[:, 1]
    fz = grads[:, 2]
    grad_mag = torch.norm(grads, dim=1)

    kg_num = fx * fx * h11 + fy * fy * h22 + fz * fz * h33
    kg_num = kg_num + 2 * h12 * fx * fy + 2 * h13 * fx * fz + 2 * h23 * fy * fz
    kg_den = grad_mag**4 + DENOM_FLOOR
    return (kg_num / kg_den).unsqueeze(1)


def compute_mean_curvature(
    dx2: torch.Tensor, dy2: torch.Tensor, dz2: torch.Tensor, grads: torch.Tensor
) -> torch.Tensor:
    """Mean curvature of the implicit surface from Hessian rows and gradients."""
    fxx = dx2[:, 0]
    fxy = dx2[:, 1]
    fxz = dx2[:, 2]
    fyy = dy2[:, 1]
    fyz = dy2[:, 2]
    fzz = dz2[:, 2]

    fx = grads[:, 0]
    fy = grads[:, 1]
    fz = grads[:, 2]
    grad_mag = torch.norm(grads, dim=1)

    km_num = fx * fx * fxx + fy * fy * fyy + fz * fz * fzz
    km_num = km_num + 2 * fxy * fx * fy + 2 * fxz * fx * fz + 2 * fyz * fy * fz
    trace_h = fxx + fyy + fzz
    km_num = km_num - grad_mag * grad_mag * trace_h
    km_den = 2 * grad_mag**3 + DENOM_FLOOR
    return (km_num / km_den).unsqueeze(1)


def compute_principal_curvatures(
    dx2: torch.Tensor,
    dy2: torch.Tensor,
    dz2: torch.Tensor,
    grads: torch.Tensor,
    epsilons: tuple[float, float, float] = _DEFAULT_CURVATURE_EPSILONS,
) -> torch.Tensor:
    """Principal curvatures with a cascading-epsilon fall-back.

    The discriminant ``km**2 - kg`` is theoretically non-negative but can dip
    below zero under floating-point error. We widen the safety floor in
    three steps; each escalation emits a single warning so the user knows
    the geometry is at the edge of numerical stability.

    Args:
        dx2: Hessian row ``∂²f/∂x∂_``.
        dy2: Hessian row ``∂²f/∂y∂_``.
        dz2: Hessian row ``∂²f/∂z∂_``.
        grads: ``(n, 3)`` field gradients.
        epsilons: Three monotonically-increasing fallback floors. Default
            covers a 28x widening between levels.

    Returns:
        ``(n, 2)`` tensor stacking ``[K1, K2]``.
    """
    kg = compute_gaussian_curvature(dx2, dy2, dz2, grads)
    km = compute_mean_curvature(dx2, dy2, dz2, grads)

    discriminant = km * km - kg
    k1 = km + torch.sqrt(discriminant + epsilons[0])
    k2 = km - torch.sqrt(discriminant + epsilons[0])

    if ((discriminant + epsilons[0]) < 0).any():
        warnings.warn(
            f"compute_principal_curvatures: discriminant below {epsilons[0]:.1e} "
            f"floor; widening to {epsilons[1]:.1e}. This indicates curvature "
            f"computation near the limit of float32 precision.",
            RuntimeWarning,
            stacklevel=2,
        )
        k1 = km + torch.sqrt(discriminant + epsilons[1])
        k2 = km - torch.sqrt(discriminant + epsilons[1])

        if ((discriminant + epsilons[1]) < 0).any():
            warnings.warn(
                f"compute_principal_curvatures: discriminant below {epsilons[1]:.1e} "
                f"floor; widening to {epsilons[2]:.1e}. Verify mesh scale and "
                f"input numerics — geometry may be effectively singular.",
                RuntimeWarning,
                stacklevel=2,
            )
            k1 = km + torch.sqrt(discriminant + epsilons[2])
            k2 = km - torch.sqrt(discriminant + epsilons[2])

    return torch.hstack((k1, k2))


def get_point_inside_mask(
    points: torch.Tensor, x_lim: float = 1.0, y_lim: float = 1.0, z_lim: float = 1.0
) -> torch.Tensor:
    """Boolean mask of points inside the normalised axis-aligned domain."""
    return (abs(points[:, 0]) < x_lim) & (abs(points[:, 1]) < y_lim) & (abs(points[:, 2]) < z_lim)


def compute_geodesic_curvature(grads1: torch.Tensor, grads2: torch.Tensor, inps: torch.Tensor) -> torch.Tensor:
    """Geodesic curvature by differentiating the tangent direction with autograd.

    Used when Hessian rows are not already available; otherwise prefer
    :func:`compute_geodesic_curvature2`, which is closed-form.
    """
    tangent = torch.cross(grads1, grads2, dim=-1)
    tangent_unit = tangent / (torch.norm(tangent, dim=1, keepdim=True) + DENOM_FLOOR)

    dtx = torch.autograd.grad(tangent_unit[:, 0], inps, torch.ones_like(tangent_unit[:, 0]), create_graph=True)[0]
    dty = torch.autograd.grad(tangent_unit[:, 1], inps, torch.ones_like(tangent_unit[:, 1]), create_graph=True)[0]
    dtz = torch.autograd.grad(tangent_unit[:, 2], inps, torch.ones_like(tangent_unit[:, 2]), create_graph=True)[0]

    accn = torch.hstack(
        (
            torch.sum(dtx * tangent_unit, dim=1, keepdim=True),
            torch.sum(dty * tangent_unit, dim=1, keepdim=True),
            torch.sum(dtz * tangent_unit, dim=1, keepdim=True),
        )
    )

    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + DENOM_FLOOR)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)


def compute_geodesic_curvature2(
    grads1: torch.Tensor,
    grads2: torch.Tensor,
    f1_h2x: torch.Tensor,
    f1_h2y: torch.Tensor,
    f1_h2z: torch.Tensor,
    f2_h2x: torch.Tensor,
    f2_h2y: torch.Tensor,
    f2_h2z: torch.Tensor,
) -> torch.Tensor:
    """Geodesic curvature from two gradients and their Hessian rows.

    Closed-form alternative to :func:`compute_geodesic_curvature` that does
    not require building an extra autograd graph through the tangent.
    """
    vector = torch.cross(grads1, grads2, dim=-1)
    vector_norm = torch.norm(vector, dim=1, keepdim=True) + DENOM_FLOOR
    tangent = vector / vector_norm

    dvx = torch.cross(f1_h2x, grads2, dim=-1) + torch.cross(grads1, f2_h2x, dim=-1)
    dvy = torch.cross(f1_h2y, grads2, dim=-1) + torch.cross(grads1, f2_h2y, dim=-1)
    dvz = torch.cross(f1_h2z, grads2, dim=-1) + torch.cross(grads1, f2_h2z, dim=-1)

    dt_x = dvx / vector_norm - vector * torch.sum(vector * dvx, dim=1, keepdim=True) / (vector_norm**3)
    dt_y = dvy / vector_norm - vector * torch.sum(vector * dvy, dim=1, keepdim=True) / (vector_norm**3)
    dt_z = dvz / vector_norm - vector * torch.sum(vector * dvz, dim=1, keepdim=True) / (vector_norm**3)

    kx = dt_x[:, 0] * tangent[:, 0] + dt_y[:, 0] * tangent[:, 1] + dt_z[:, 0] * tangent[:, 2]
    ky = dt_x[:, 1] * tangent[:, 0] + dt_y[:, 1] * tangent[:, 1] + dt_z[:, 1] * tangent[:, 2]
    kz = dt_x[:, 2] * tangent[:, 0] + dt_y[:, 2] * tangent[:, 1] + dt_z[:, 2] * tangent[:, 2]

    accn = torch.hstack((kx.unsqueeze(1), ky.unsqueeze(1), kz.unsqueeze(1)))
    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + DENOM_FLOOR)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)
