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
    can be jointly optimised with the scalar field. ``get_platform_pos_loss``
    sums four constraints:

    - direction error against locally-detected build directions,
    - signed distance of the part above the platform plane,
    - outer-ring clearance check, and
    - inner-ring clearance check.
    """

    def __init__(self, device: str = "cuda", scale: float = 1.0) -> None:
        super().__init__()
        self.device = device
        self.platform_base = nn.Parameter(torch.tensor([[0, 0, 0]], dtype=torch.float32, device=self.device))
        self.platform_dir = nn.Parameter(torch.tensor([[0, 1, 0]], dtype=torch.float32, device=self.device))
        self.selected_points: torch.Tensor | None = None
        self.target_dirs: torch.Tensor | None = None
        self.disp_dist: float = 5.00 / scale
        self.disp_dist2: float = 5.00 / scale
        self.scale: float = scale

    def select_points(
        self,
        surface_points: torch.Tensor,
        surface_grads: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> None:
        """Identify boundary points whose layer-gradient direction needs platform support.

        Sets ``self.selected_points`` and ``self.target_dirs`` in-place.
        """
        grad_norm = torch.norm(surface_grads, dim=1).unsqueeze(1)
        surface_grads = surface_grads / (grad_norm + DENOM_FLOOR)
        dot_prod = torch.sum(surface_normals * surface_grads, dim=1)

        support_error = -dot_prod + np.cos(np.deg2rad(_PLATFORM_SUPPORT_ANGLE_DEG))
        support_mask = torch.relu(support_error) > 0.0

        self.selected_points = surface_points[support_mask].detach()
        self.selected_points.requires_grad = True
        self.target_dirs = surface_grads[support_mask].detach()

    def sample_circle_around_gradient(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        k: int,
        axial_distance: float,
        radius: float,
    ) -> torch.Tensor:
        """Sample clearance-check points on a circle around each gradient axis.

        Args:
            points: ``(n, 3)`` base points.
            grads: ``(n, 3)`` gradient at each base point.
            k: Samples per circle.
            axial_distance: Distance along the gradient at which to place
                the circle centre.
            radius: Circle radius.

        Returns:
            ``(n, k, 3)`` sampled points.
        """
        axis = grads / (grads.norm(dim=1, keepdim=True) + DENOM_FLOOR)
        axial_points = points + axial_distance * axis

        rand_vec = torch.randn(points.shape[0], 3, device=self.device)
        rand_vec = rand_vec - (rand_vec * axis).sum(dim=1, keepdim=True) * axis
        radial1 = rand_vec / (rand_vec.norm(dim=1, keepdim=True) + DENOM_FLOOR)
        radial2 = torch.cross(axis, radial1, dim=-1)

        angles = torch.linspace(0, 2 * torch.pi, steps=k, device=self.device)
        cosines = torch.cos(angles).view(1, k, 1)
        sines = torch.sin(angles).view(1, k, 1)

        radial_dirs = cosines * radial1.unsqueeze(1) + sines * radial2.unsqueeze(1)
        return axial_points.unsqueeze(1) + radius * radial_dirs

    def get_platform_pos_loss(
        self,
        surface_points: torch.Tensor,
        surface_grads: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> torch.Tensor:
        """Total platform loss: direction + distance + outer/inner clearance."""
        self.select_points(surface_points, surface_grads, surface_normals)

        platform_dir_norm = torch.norm(self.platform_dir) + DENOM_FLOOR

        if self.target_dirs is not None:
            dir_error = self.target_dirs - (self.platform_dir / platform_dir_norm)
            dir_loss = torch.mean(dir_error * dir_error)
        else:
            dir_loss = torch.zeros((), device=surface_points.device)

        if self.selected_points is not None:
            disp_vector = surface_points.detach() - self.platform_base
            disps = torch.sum(disp_vector * self.platform_dir / platform_dir_norm, dim=1)
            disps = disps - self.disp_dist
            disp_error2 = torch.min(disps)
            disp_loss2 = torch.mean(disp_error2 * disp_error2)
        else:
            disp_loss2 = torch.zeros((), device=surface_points.device)

        check_samples = self.sample_circle_around_gradient(
            surface_points, surface_grads, 6, 60 / self.scale, 25 / self.scale
        )
        clear_loss_outer = self._platform_clearance_loss(check_samples, platform_dir_norm)

        check_samples = self.sample_circle_around_gradient(
            surface_points, surface_grads, 6, 30 / self.scale, 20 / self.scale
        )
        clear_loss_inner = self._platform_clearance_loss(check_samples, platform_dir_norm)

        return dir_loss + _W_DISP2 * disp_loss2 + _W_CLEAR_OUTER * clear_loss_outer + _W_CLEAR_INNER * clear_loss_inner

    def _platform_clearance_loss(self, check_samples: torch.Tensor, platform_dir_norm: torch.Tensor) -> torch.Tensor:
        """Penalise sampled points that fall inside the platform clearance band."""
        check_samples = check_samples.view(-1, 3)
        disp_vector = check_samples.detach() - self.platform_base
        disps = torch.sum(disp_vector * self.platform_dir / platform_dir_norm, dim=1)
        disps = self.disp_dist2 - disps
        disp_error = _PLATFORM_CLEARANCE_RELU_SCALING * torch.relu(disps)
        error_mask = disp_error > 0
        if torch.sum(error_mask) < 1:
            return torch.zeros((), device=check_samples.device)
        return torch.mean(disp_error[error_mask] * disp_error[error_mask])


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
    platform_loss = platform_model.get_platform_pos_loss(boundary_points, boundary_outs["grads"], boundary_normals)
    return loss + platform_loss, platform_loss
