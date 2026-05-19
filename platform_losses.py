"""Platform model and platform-clearance loss for alignment examples."""

import numpy as np
import torch
import torch.nn as nn

from training_dataclasses import loss_enabled


class PlatformModel(nn.Module):
    """Small trainable platform pose model used by platform loss."""

    def __init__(self, device="cuda", scale=1.0):
        super().__init__()
        self.device = device
        self.platformBase = nn.Parameter(torch.tensor([[0, 0, 0]], dtype=torch.float32, device=self.device))
        self.platformDir = nn.Parameter(torch.tensor([[0, 1, 0]], dtype=torch.float32, device=self.device))
        self.selectedPoints = None
        self.targetDirs = None
        self.dispDist = 5.00 / scale
        self.dispDist2 = 5.00 / scale
        self.scale = scale

    def selectPoints(self, surfacePoints, surfaceGrads, surfaceNormals):
        """Select boundary points whose layer gradients need platform support."""
        gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
        surfaceGrads = surfaceGrads / (gradNorm + 1e-10)
        dotProd = surfaceNormals * surfaceGrads
        dotProd = torch.sum(dotProd, dim=1)

        supportError = -dotProd + np.cos(137.00 * 3.1457 / 180.00)
        supportMask = torch.relu(supportError) > 0.0

        self.selectedPoints = surfacePoints[supportMask].detach()
        self.selectedPoints.requires_grad = True
        self.targetDirs = surfaceGrads[supportMask].detach()

    def sample_circle_around_gradient(self, points, grads, k, l, r):
        """Sample clearance-check points on circles around each gradient axis."""
        axis = grads / (grads.norm(dim=1, keepdim=True) + 1e-8)
        axial_points = points + l * axis

        rand_vec = torch.randn(points.shape[0], 3, device=self.device)
        rand_vec = rand_vec - (rand_vec * axis).sum(dim=1, keepdim=True) * axis
        radial1 = rand_vec / (rand_vec.norm(dim=1, keepdim=True) + 1e-8)
        radial2 = torch.cross(axis, radial1)

        angles = torch.linspace(0, 2 * torch.pi, steps=k, device=self.device)
        cosines = torch.cos(angles).view(1, k, 1)
        sines = torch.sin(angles).view(1, k, 1)

        radial_dirs = cosines * radial1.unsqueeze(1) + sines * radial2.unsqueeze(1)
        return axial_points.unsqueeze(1) + r * radial_dirs

    def getPlatformPosLoss(self, surfacePoints, surfaceGrads, surfaceNormals):
        """Penalize platform direction, distance, and clearance violations."""
        self.selectPoints(surfacePoints, surfaceGrads, surfaceNormals)

        palformDirNorm = torch.norm(self.platformDir) + 1e-10

        if self.targetDirs is not None:
            dirError = self.targetDirs - (self.platformDir / palformDirNorm)
            dirLoss = torch.mean(dirError * dirError)
        else:
            dirLoss = 0

        if self.selectedPoints is not None:
            dispVector = surfacePoints.detach() - self.platformBase
            disps = torch.sum(dispVector * self.platformDir / palformDirNorm, dim=1)
            disps = self.dispDist - disps
            dispError = torch.relu(disps)
            errorMask = dispError > 0
            dispLoss = torch.mean(dispError[errorMask] * dispError[errorMask])
            if torch.sum(errorMask) < 1:
                dispLoss = 0
        else:
            dispLoss = 0

        if self.selectedPoints is not None:
            dispVector = surfacePoints.detach() - self.platformBase
            disps = torch.sum(dispVector * self.platformDir / palformDirNorm, dim=1)
            disps = disps - self.dispDist
            dispError2 = torch.min(disps)
            dispLoss2 = torch.mean(dispError2 * dispError2)
        else:
            dispLoss2 = 0

        checkSamples = self.sample_circle_around_gradient(surfacePoints, surfaceGrads, 6, 60 / self.scale, 25 / self.scale)
        dispLossCol = self._platform_clearance_loss(checkSamples, palformDirNorm)

        checkSamples = self.sample_circle_around_gradient(surfacePoints, surfaceGrads, 6, 30 / self.scale, 20 / self.scale)
        dispLossCol2 = self._platform_clearance_loss(checkSamples, palformDirNorm)

        return dirLoss + 0.1 * dispLoss2 + 0.05 * dispLossCol + 0.05 * dispLossCol2

    def _platform_clearance_loss(self, checkSamples, palformDirNorm):
        """Penalize sampled points that fall inside the platform clearance band."""
        checkSamples = checkSamples.view(-1, 3)
        dispVector = checkSamples.detach() - self.platformBase
        disps = torch.sum(dispVector * self.platformDir / palformDirNorm, dim=1)
        disps = self.dispDist2 - disps
        dispError = 10 * torch.relu(disps)
        errorMask = dispError > 0
        if torch.sum(errorMask) < 1:
            return 0
        return torch.mean(dispError[errorMask] * dispError[errorMask])


def add_platform_loss(loss, platform_model, scalar_field, boundary_points, boundary_normals, epoch, config):
    """Add platform loss after the configured start epoch."""
    if epoch <= config.platform_start_epoch or not loss_enabled(config, "use_platform_loss"):
        return loss, 0

    boundary_outs = scalar_field(boundary_points)
    platform_loss = platform_model.getPlatformPosLoss(boundary_points, boundary_outs["grads"], boundary_normals)
    return loss + platform_loss, platform_loss
