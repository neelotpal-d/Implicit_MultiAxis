"""Differential-geometry helpers used by the layer / toolpath losses.

Function naming preserves the original camelCase public API; downstream
modules import these by name, and changing them would touch the example
pipelines and the displays without changing behaviour. The functions
operate on PyTorch tensors and return PyTorch tensors so they can be
composed into autograd graphs.
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


def supportLoss(
    surfaceNormals: torch.Tensor,
    surfaceGrads: torch.Tensor,
    angle_degrees: float = 132.0,
    sharpness: float = 25.0,
) -> dict:
    """Penalise boundary gradients that violate the support-angle threshold.

    Args:
        surfaceNormals: ``(n, 3)`` outward normals at boundary points.
        surfaceGrads: ``(n, 3)`` field gradients at the same points.
        angle_degrees: Maximum allowed angle between the build direction
            and the negative surface normal. Default 132° matches the
            published experiments.
        sharpness: Multiplier applied inside the hinge, before the square.

    Returns:
        ``{"loss": scalar, "mask": (n,) bool tensor of violation points}``.
    """
    gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
    surfaceGrads = surfaceGrads / (gradNorm + DENOM_FLOOR)
    dotProd = torch.sum(surfaceNormals * surfaceGrads, dim=1)

    supportError = -dotProd + np.cos(np.deg2rad(angle_degrees))
    supportError = torch.relu(sharpness * supportError)
    supportMask = supportError > 0.0
    loss = torch.mean(supportError * supportError)
    return {"loss": loss, "mask": supportMask}


def computeGaussianCurvature(
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
    norm_gradF = torch.norm(grads, dim=1)

    Kg_num = fx * fx * h11 + fy * fy * h22 + fz * fz * h33
    Kg_num = Kg_num + 2 * h12 * fx * fy + 2 * h13 * fx * fz + 2 * h23 * fy * fz
    Kg_den = norm_gradF * norm_gradF * norm_gradF * norm_gradF + DENOM_FLOOR
    return (Kg_num / Kg_den).unsqueeze(1)


def computeMeanCurvature(dx2: torch.Tensor, dy2: torch.Tensor, dz2: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
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
    norm_gradF = torch.norm(grads, dim=1)

    Km_num = fx * fx * fxx + fy * fy * fyy + fz * fz * fzz
    Km_num = Km_num + 2 * fxy * fx * fy + 2 * fxz * fx * fz + 2 * fyz * fy * fz
    trace_h = fxx + fyy + fzz
    Km_num = Km_num - norm_gradF * norm_gradF * trace_h
    Km_den = 2 * norm_gradF * norm_gradF * norm_gradF + DENOM_FLOOR
    return (Km_num / Km_den).unsqueeze(1)


def computePrincipalCurvatures(
    dx2: torch.Tensor,
    dy2: torch.Tensor,
    dz2: torch.Tensor,
    grads: torch.Tensor,
    epsilons: tuple[float, float, float] = _DEFAULT_CURVATURE_EPSILONS,
) -> torch.Tensor:
    """Principal curvatures with a cascading-epsilon fall-back.

    The discriminant ``Km**2 - Kg`` is theoretically non-negative but can dip
    below zero under floating-point error. We widen the safety floor in
    three steps; each escalation emits a single warning so the user knows
    the geometry is at the edge of numerical stability.

    Args:
        dx2, dy2, dz2: Hessian rows.
        grads: ``(n, 3)`` field gradients.
        epsilons: Three monotonically-increasing fallback floors. Default
            covers a 28x widening between levels.

    Returns:
        ``(n, 2)`` tensor stacking ``[K1, K2]``.
    """
    Kg = computeGaussianCurvature(dx2, dy2, dz2, grads)
    Km = computeMeanCurvature(dx2, dy2, dz2, grads)

    discriminant = Km * Km - Kg
    K1 = Km + torch.sqrt(discriminant + epsilons[0])
    K2 = Km - torch.sqrt(discriminant + epsilons[0])

    if ((discriminant + epsilons[0]) < 0).any():
        warnings.warn(
            f"computePrincipalCurvatures: discriminant below {epsilons[0]:.1e} "
            f"floor; widening to {epsilons[1]:.1e}. This indicates curvature "
            f"computation near the limit of float32 precision.",
            RuntimeWarning,
            stacklevel=2,
        )
        K1 = Km + torch.sqrt(discriminant + epsilons[1])
        K2 = Km - torch.sqrt(discriminant + epsilons[1])

        if ((discriminant + epsilons[1]) < 0).any():
            warnings.warn(
                f"computePrincipalCurvatures: discriminant below {epsilons[1]:.1e} "
                f"floor; widening to {epsilons[2]:.1e}. Verify mesh scale and "
                f"input numerics — geometry may be effectively singular.",
                RuntimeWarning,
                stacklevel=2,
            )
            K1 = Km + torch.sqrt(discriminant + epsilons[2])
            K2 = Km - torch.sqrt(discriminant + epsilons[2])

    return torch.hstack((K1, K2))


def getPointInsideMask(
    points: torch.Tensor, x_lim: float = 1.0, y_lim: float = 1.0, z_lim: float = 1.0
) -> torch.Tensor:
    """Boolean mask of points inside the normalised axis-aligned domain."""
    return (abs(points[:, 0]) < x_lim) & (abs(points[:, 1]) < y_lim) & (abs(points[:, 2]) < z_lim)


def computeGeodesicCurvature(grads1: torch.Tensor, grads2: torch.Tensor, inps: torch.Tensor) -> torch.Tensor:
    """Geodesic curvature by differentiating the tangent direction with autograd.

    Used when Hessian rows are not already available; otherwise prefer
    :func:`computeGeodesicCurvature2`, which is closed-form.
    """
    tangent = torch.cross(grads1, grads2, dim=-1)
    tangent_unit = tangent / (torch.norm(tangent, dim=1, keepdim=True) + DENOM_FLOOR)

    dTx = torch.autograd.grad(tangent_unit[:, 0], inps, torch.ones_like(tangent_unit[:, 0]), create_graph=True)[0]
    dTy = torch.autograd.grad(tangent_unit[:, 1], inps, torch.ones_like(tangent_unit[:, 1]), create_graph=True)[0]
    dTz = torch.autograd.grad(tangent_unit[:, 2], inps, torch.ones_like(tangent_unit[:, 2]), create_graph=True)[0]

    accn = torch.hstack(
        (
            torch.sum(dTx * tangent_unit, dim=1, keepdim=True),
            torch.sum(dTy * tangent_unit, dim=1, keepdim=True),
            torch.sum(dTz * tangent_unit, dim=1, keepdim=True),
        )
    )

    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + DENOM_FLOOR)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)


def computeGeodesicCurvature2(
    grads1: torch.Tensor,
    grads2: torch.Tensor,
    f1H2X: torch.Tensor,
    f1H2Y: torch.Tensor,
    f1H2Z: torch.Tensor,
    f2H2X: torch.Tensor,
    f2H2Y: torch.Tensor,
    f2H2Z: torch.Tensor,
) -> torch.Tensor:
    """Geodesic curvature from two gradients and their Hessian rows.

    Closed-form alternative to :func:`computeGeodesicCurvature` that does
    not require building an extra autograd graph through the tangent.
    """
    vector = torch.cross(grads1, grads2, dim=-1)
    vector_norm = torch.norm(vector, dim=1, keepdim=True) + DENOM_FLOOR
    tangent = vector / vector_norm

    dVx = torch.cross(f1H2X, grads2, dim=-1) + torch.cross(grads1, f2H2X, dim=-1)
    dVy = torch.cross(f1H2Y, grads2, dim=-1) + torch.cross(grads1, f2H2Y, dim=-1)
    dVz = torch.cross(f1H2Z, grads2, dim=-1) + torch.cross(grads1, f2H2Z, dim=-1)

    dT_x = dVx / vector_norm - vector * (torch.sum(vector * dVx, dim=1, keepdim=True)) / (vector_norm**3)
    dT_y = dVy / vector_norm - vector * (torch.sum(vector * dVy, dim=1, keepdim=True)) / (vector_norm**3)
    dT_z = dVz / vector_norm - vector * (torch.sum(vector * dVz, dim=1, keepdim=True)) / (vector_norm**3)

    Kx = dT_x[:, 0] * tangent[:, 0] + dT_y[:, 0] * tangent[:, 1] + dT_z[:, 0] * tangent[:, 2]
    Ky = dT_x[:, 1] * tangent[:, 0] + dT_y[:, 1] * tangent[:, 1] + dT_z[:, 1] * tangent[:, 2]
    Kz = dT_x[:, 2] * tangent[:, 0] + dT_y[:, 2] * tangent[:, 1] + dT_z[:, 2] * tangent[:, 2]

    accn = torch.hstack((Kx.unsqueeze(1), Ky.unsqueeze(1), Kz.unsqueeze(1)))
    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + DENOM_FLOOR)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)
