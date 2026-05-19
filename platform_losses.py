"""Platform model and platform-clearance loss for alignment examples."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from constants import DENOM_FLOOR
from training_dataclasses import loss_enabled

_PLATFORM_SUPPORT_ANGLE_DEG: float = 137.0
"""Support-angle threshold (deg) used to select boundary points whose
layer gradient needs the platform's mechanical support."""

_PLATFORM_CLEARANCE_RELU_SCALING: float = 10.0
"""Multiplier on the platform clearance violation before the relu hinge."""

# Coefficients combining the four platform-loss components into a single
# scalar. Pre-existing tuning from the original experiments.
_W_DISP2: float = 0.1
_W_CLEAR_OUTER: float = 0.05
_W_CLEAR_INNER: float = 0.05


class PlatformModel(nn.Module):
    """Trainable build-platform pose used by the platform-clearance loss.

    The platform is modelled as a half-space, parameterised by a base
    point and outward direction. Both are :class:`nn.Parameter` so they
    can be jointly optimised with the scalar field. ``getPlatformPosLoss``
    sums four constraints:

    - direction error against locally-detected build directions,
    - signed distance of the part above the platform plane,
    - outer-ring clearance check, and
    - inner-ring clearance check.
    """

    def __init__(self, device: str = "cuda", scale: float = 1.0) -> None:
        super().__init__()
        self.device = device
        self.platformBase = nn.Parameter(torch.tensor([[0, 0, 0]], dtype=torch.float32, device=self.device))
        self.platformDir = nn.Parameter(torch.tensor([[0, 1, 0]], dtype=torch.float32, device=self.device))
        self.selectedPoints: torch.Tensor | None = None
        self.targetDirs: torch.Tensor | None = None
        self.dispDist: float = 5.00 / scale
        self.dispDist2: float = 5.00 / scale
        self.scale: float = scale

    def selectPoints(
        self,
        surfacePoints: torch.Tensor,
        surfaceGrads: torch.Tensor,
        surfaceNormals: torch.Tensor,
    ) -> None:
        """Identify boundary points whose layer-gradient direction needs platform support.

        Sets ``self.selectedPoints`` and ``self.targetDirs`` in-place.
        """
        gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
        surfaceGrads = surfaceGrads / (gradNorm + DENOM_FLOOR)
        dotProd = torch.sum(surfaceNormals * surfaceGrads, dim=1)

        supportError = -dotProd + np.cos(np.deg2rad(_PLATFORM_SUPPORT_ANGLE_DEG))
        supportMask = torch.relu(supportError) > 0.0

        self.selectedPoints = surfacePoints[supportMask].detach()
        self.selectedPoints.requires_grad = True
        self.targetDirs = surfaceGrads[supportMask].detach()

    def sample_circle_around_gradient(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        k: int,
        l: float,
        r: float,
    ) -> torch.Tensor:
        """Sample clearance-check points on a circle around each gradient axis.

        Args:
            points: ``(n, 3)`` base points.
            grads: ``(n, 3)`` gradient at each base point.
            k: Samples per circle.
            l: Distance along the gradient at which to place the circle centre.
            r: Circle radius.

        Returns:
            ``(n, k, 3)`` sampled points.
        """
        axis = grads / (grads.norm(dim=1, keepdim=True) + DENOM_FLOOR)
        axial_points = points + l * axis

        rand_vec = torch.randn(points.shape[0], 3, device=self.device)
        rand_vec = rand_vec - (rand_vec * axis).sum(dim=1, keepdim=True) * axis
        radial1 = rand_vec / (rand_vec.norm(dim=1, keepdim=True) + DENOM_FLOOR)
        radial2 = torch.cross(axis, radial1, dim=-1)

        angles = torch.linspace(0, 2 * torch.pi, steps=k, device=self.device)
        cosines = torch.cos(angles).view(1, k, 1)
        sines = torch.sin(angles).view(1, k, 1)

        radial_dirs = cosines * radial1.unsqueeze(1) + sines * radial2.unsqueeze(1)
        return axial_points.unsqueeze(1) + r * radial_dirs

    def getPlatformPosLoss(
        self,
        surfacePoints: torch.Tensor,
        surfaceGrads: torch.Tensor,
        surfaceNormals: torch.Tensor,
    ) -> torch.Tensor:
        """Total platform loss: direction + distance + outer/inner clearance."""
        self.selectPoints(surfacePoints, surfaceGrads, surfaceNormals)

        platformDirNorm = torch.norm(self.platformDir) + DENOM_FLOOR

        if self.targetDirs is not None:
            dirError = self.targetDirs - (self.platformDir / platformDirNorm)
            dirLoss = torch.mean(dirError * dirError)
        else:
            dirLoss = torch.zeros((), device=surfacePoints.device)

        if self.selectedPoints is not None:
            dispVector = surfacePoints.detach() - self.platformBase
            disps = torch.sum(dispVector * self.platformDir / platformDirNorm, dim=1)
            disps = disps - self.dispDist
            dispError2 = torch.min(disps)
            dispLoss2 = torch.mean(dispError2 * dispError2)
        else:
            dispLoss2 = torch.zeros((), device=surfacePoints.device)

        checkSamples = self.sample_circle_around_gradient(
            surfacePoints, surfaceGrads, 6, 60 / self.scale, 25 / self.scale
        )
        dispLossCol = self._platform_clearance_loss(checkSamples, platformDirNorm)

        checkSamples = self.sample_circle_around_gradient(
            surfacePoints, surfaceGrads, 6, 30 / self.scale, 20 / self.scale
        )
        dispLossCol2 = self._platform_clearance_loss(checkSamples, platformDirNorm)

        return dirLoss + _W_DISP2 * dispLoss2 + _W_CLEAR_OUTER * dispLossCol + _W_CLEAR_INNER * dispLossCol2

    def _platform_clearance_loss(self, checkSamples: torch.Tensor, platformDirNorm: torch.Tensor) -> torch.Tensor:
        """Penalise sampled points that fall inside the platform clearance band."""
        checkSamples = checkSamples.view(-1, 3)
        dispVector = checkSamples.detach() - self.platformBase
        disps = torch.sum(dispVector * self.platformDir / platformDirNorm, dim=1)
        disps = self.dispDist2 - disps
        dispError = _PLATFORM_CLEARANCE_RELU_SCALING * torch.relu(disps)
        errorMask = dispError > 0
        if torch.sum(errorMask) < 1:
            return torch.zeros((), device=checkSamples.device)
        return torch.mean(dispError[errorMask] * dispError[errorMask])


def add_platform_loss(
    loss: torch.Tensor,
    platform_model: PlatformModel,
    scalar_field,
    boundary_points: torch.Tensor,
    boundary_normals: torch.Tensor,
    epoch: int,
    config,
) -> tuple[torch.Tensor, torch.Tensor | int]:
    """Add platform loss after the configured start epoch."""
    if epoch <= config.platform_start_epoch or not loss_enabled(config, "use_platform_loss"):
        return loss, 0

    boundary_outs = scalar_field(boundary_points)
    platform_loss = platform_model.getPlatformPosLoss(boundary_points, boundary_outs["grads"], boundary_normals)
    return loss + platform_loss, platform_loss
