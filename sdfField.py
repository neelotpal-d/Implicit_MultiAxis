"""SIREN-based signed-distance-field model utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import igl
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader, TensorDataset

from siren_pytorch import SirenNet


@dataclass(frozen=True)
class SDFLossWeights:
    """Weights used when fitting the signed-distance approximation."""

    distance_scalar: float = 1e2
    surface_scalar: float = 1e2
    surface_normal: float = 1e0
    eikonal: float = 1e-1


@dataclass(frozen=True)
class SDFTrainingConfig:
    """Training controls for the SDF model."""

    device: str = "cuda"
    learning_rate: float = 1e-5
    epoch_count: int = 60
    batch_size: int = 3000
    save_every_epochs: int = 5
    save_path: str = "./outSDF.pt"
    weights: SDFLossWeights = field(default_factory=SDFLossWeights)


def build_sdf_network(device="cuda"):
    """Create the SIREN network used for SDF prediction."""
    return SirenNet(3, 256, 1, 5, w0=30, w0_initial=30).to(device)


def save_sdf_checkpoint(model, save_path):
    """Save only the model state dict used by the collision utilities."""
    torch.save({"model_state_dict": model.state_dict()}, save_path)


def load_sdf_checkpoint(model, checkpoint_path, device="cuda"):
    """Load an SDF checkpoint into an existing model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_inside_outside_targets(sdf_values, device="cuda"):
    """
    Build sign targets for optional classification experiments.

    Currently this is not used by the training loop because the classification
    loss was disabled in the original workflow.
    """
    targets = torch.ones_like(sdf_values.unsqueeze(1), device=device)
    targets[sdf_values > 0] = -1
    return targets


def ensure_requires_grad(points):
    """Return a detached tensor with gradients enabled."""
    points = points.detach()
    points.requires_grad = True
    return points


def make_sdf_loaders(points_dist, scalars_dist, points_surf, normals_surf, batch_size):
    """Create paired data loaders for distance samples and surface samples."""
    dist_dataset = TensorDataset(points_dist, scalars_dist)
    surf_dataset = TensorDataset(points_surf, normals_surf)

    dist_loader = DataLoader(dist_dataset, batch_size=batch_size, shuffle=True)
    surf_loader = DataLoader(surf_dataset, batch_size=batch_size, shuffle=True)

    if len(surf_loader) < len(dist_loader):
        surf_loader = cycle(surf_loader)

    return dist_loader, surf_loader


def compute_distance_losses(model, points, target_scalars, weights):
    """Compute scalar and eikonal losses away from the surface."""
    out = model(points)
    scalar_error = torch.abs(out["scalars"] - target_scalars.unsqueeze(1))
    scalar_loss = torch.mean(scalar_error)

    norm_error = 1 - torch.norm(out["grads"], dim=1)
    eikonal_loss = torch.mean(norm_error * norm_error)

    total = weights.distance_scalar * scalar_loss + weights.eikonal * eikonal_loss
    return total, {"distance_scalar": scalar_loss, "eikonal": eikonal_loss}


def compute_surface_losses(model, points, target_normals, weights):
    """Compute zero-level and normal-alignment losses on surface points."""
    out = model(points)
    normal_error = out["grads"] - target_normals
    normal_loss = torch.mean(normal_error * normal_error)

    scalar_loss = torch.mean(out["scalars"] * out["scalars"])
    total = weights.surface_normal * normal_loss + weights.surface_scalar * scalar_loss
    return total, {"surface_normal": normal_loss, "surface_scalar": scalar_loss}


def train_sdf_model(
    model,
    points_dist,
    scalars_dist,
    points_surf,
    normals_surf,
    config=None,
):
    """Fit an SDF model from distance samples and surface normals."""
    config = config or SDFTrainingConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    points_dist = ensure_requires_grad(points_dist)
    points_surf = ensure_requires_grad(points_surf)
    dist_loader, surf_loader = make_sdf_loaders(
        points_dist,
        scalars_dist,
        points_surf,
        normals_surf,
        config.batch_size,
    )

    for epoch in range(config.epoch_count):
        epoch_loss = 0.0
        distance_scalar_loss = 0.0
        surface_scalar_loss = 0.0
        batch_count = 0

        for (batch_points_dist, batch_scalars_dist), (batch_points_surf, batch_normals_surf) in zip(
            dist_loader,
            surf_loader,
        ):
            optimizer.zero_grad()

            distance_loss, distance_terms = compute_distance_losses(
                model,
                batch_points_dist,
                batch_scalars_dist,
                config.weights,
            )
            surface_loss, surface_terms = compute_surface_losses(
                model,
                batch_points_surf,
                batch_normals_surf,
                config.weights,
            )

            loss = distance_loss + surface_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            distance_scalar_loss += float(distance_terms["distance_scalar"].detach())
            surface_scalar_loss += float(surface_terms["surface_scalar"].detach())
            batch_count += 1

        epoch_loss /= max(batch_count, 1)
        distance_scalar_loss /= max(batch_count, 1)
        surface_scalar_loss /= max(batch_count, 1)
        print(
            f"epoch {epoch + 1}/{config.epoch_count} "
            f"loss={epoch_loss:.6g} "
            f"distance_scalar={distance_scalar_loss:.6g} "
            f"surface_scalar={surface_scalar_loss:.6g}"
        )

        if config.save_every_epochs and (epoch + 1) % config.save_every_epochs == 0:
            save_sdf_checkpoint(model, config.save_path)

    return model


def predict_scalars(model, points, batch_size=100):
    """Predict SDF scalar values in batches."""
    outputs = []
    for start in range(0, points.shape[0], batch_size):
        batch = points[start : start + batch_size]
        outputs.append(model(batch)["scalars"])
    return {"scalars": torch.cat(outputs, dim=0)}


def predict_grads(model, points):
    """Predict SDF gradients."""
    return {"grads": model(points)["grads"]}


def mesh_bounds(mesh):
    """Return mesh vertices, triangular faces, and axis-aligned bounds."""
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:4]
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return vertices, faces, mins, maxs


def normalization_tensors(mins, maxs, device="cuda"):
    """Build center and isotropic half-range tensors for normalized coordinates."""
    mins = torch.tensor(mins, dtype=torch.float32, device=device)
    maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
    mid_vals = 0.5 * (mins + maxs)
    max_range = torch.max(maxs - mins)
    range_vals = 0.5 * torch.ones(3, dtype=torch.float32, device=device) * max_range
    return mid_vals, range_vals


def build_sdf_training_samples(mesh, padding=2.0, spacing=0.2, device="cuda"):
    """Create normalized distance samples and normalized surface samples from a mesh."""
    vertices, faces, mins, maxs = mesh_bounds(mesh)
    x_vals = np.arange(mins[0] - padding, maxs[0] + padding, spacing)
    y_vals = np.arange(mins[1] - padding, maxs[1] + padding, spacing)
    z_vals = np.arange(mins[2] - padding, maxs[2] + padding, spacing)
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)
    grid = pv.StructuredGrid(x_grid.flatten(), y_grid.flatten(), z_grid.flatten())

    sdf = igl.signed_distance(grid.points, vertices, faces)[0]
    mid_vals, range_vals = normalization_tensors(mins, maxs, device=device)
    scale = float(range_vals[0].detach().to("cpu"))

    points_dist = torch.tensor(grid.points, dtype=torch.float32, device=device)
    scalars_dist = torch.tensor(sdf / scale, dtype=torch.float32, device=device)
    points_dist = ensure_requires_grad((points_dist - mid_vals) / range_vals)

    mesh.compute_normals(cell_normals=False, inplace=True)
    points_surf = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    normals_surf = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32, device=device)
    normals_surf = normals_surf / torch.norm(normals_surf, dim=1, keepdim=True)
    points_surf = ensure_requires_grad((points_surf - mid_vals) / range_vals)

    return points_dist, scalars_dist, points_surf, normals_surf, mid_vals, range_vals


def sample_points_near_surface(mesh, distance_scale=0.02):
    """Sample points offset from mesh vertices along vertex normals."""
    mesh = mesh.copy()
    mesh.compute_normals(cell_normals=False, inplace=True)

    points = torch.tensor(mesh.points, dtype=torch.float32)
    normals = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    distances = distance_scale * torch.rand(points.shape[0])
    return points + distances.unsqueeze(1) * normals, normals


def limit_samples(points, scalars, max_count):
    """Limit distance samples before training."""
    if max_count is None or points.shape[0] <= max_count:
        return points, scalars
    return points[:max_count], scalars[:max_count]


def train_sdf_from_mesh(
    mesh_path,
    save_path,
    device="cuda",
    learning_rate=1e-5,
    epoch_count=60,
    batch_size=3000,
    save_every_epochs=5,
    padding=2.0,
    spacing=0.2,
    max_distance_points=None,
):
    """Train an SDF checkpoint directly from a mesh file."""
    mesh = pv.read(mesh_path)
    points_dist, scalars_dist, points_surf, normals_surf, _, _ = build_sdf_training_samples(
        mesh,
        padding=padding,
        spacing=spacing,
        device=device,
    )
    points_dist, scalars_dist = limit_samples(points_dist, scalars_dist, max_distance_points)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model = build_sdf_network(device)
    config = SDFTrainingConfig(
        device=device,
        learning_rate=learning_rate,
        epoch_count=epoch_count,
        batch_size=batch_size,
        save_every_epochs=save_every_epochs,
        save_path=save_path,
    )
    train_sdf_model(model, points_dist, scalars_dist, points_surf, normals_surf, config)
    save_sdf_checkpoint(model, save_path)
    return model


class SDFModel:
    """Small wrapper around the SDF SIREN used by collision losses."""

    def __init__(self, device: str = "cuda", model_load_path: str | None = None) -> None:
        self.device = device
        self.model = build_sdf_network(device)
        if model_load_path:
            load_sdf_checkpoint(self.model, model_load_path, device=device)

    def train(
        self,
        points_dist: torch.Tensor,
        scalars_dist: torch.Tensor,
        points_surf: torch.Tensor,
        normals_surf: torch.Tensor,
        lr: float = 1e-5,
        epoch_count: int = 60,
        save_path: str = "./outSDF.pt",
        batch_size: int = 3000,
    ) -> torch.nn.Module:
        """Train the wrapped SDF model from distance + surface samples."""
        config = SDFTrainingConfig(
            device=self.device,
            learning_rate=lr,
            epoch_count=epoch_count,
            batch_size=batch_size,
            save_path=save_path,
        )
        return train_sdf_model(
            self.model,
            points_dist,
            scalars_dist,
            points_surf,
            normals_surf,
            config,
        )

    def predict_outs(self, points: torch.Tensor, batch_size: int = 100) -> dict:
        """Batched SDF scalar predictions."""
        return predict_scalars(self.model, points, batch_size=batch_size)

    def predict_grads(self, points: torch.Tensor, batch_size: int = 100) -> dict:
        """SDF gradient predictions."""
        return predict_grads(self.model, points)


def parse_args():
    """Parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an SDF checkpoint from a mesh.")
    train_parser.add_argument("mesh_path", help="Surface mesh path readable by PyVista.")
    train_parser.add_argument("--save-path", default="./outSDF.pt", help="Output checkpoint path.")
    train_parser.add_argument("--device", default="cuda", help="Torch device, such as cuda or cpu.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Adam learning rate.")
    train_parser.add_argument("--epochs", type=int, default=60, help="Training epoch count.")
    train_parser.add_argument("--batch-size", type=int, default=3000, help="Training batch size.")
    train_parser.add_argument("--save-every", type=int, default=5, help="Checkpoint write interval in epochs.")
    train_parser.add_argument("--padding", type=float, default=2.0, help="Grid padding around mesh bounds.")
    train_parser.add_argument("--spacing", type=float, default=0.2, help="Distance-sample grid spacing.")
    train_parser.add_argument(
        "--max-distance-points",
        type=int,
        default=None,
        help="Optional cap on distance samples used for training.",
    )
    return parser.parse_args()


def main():
    """Run the command-line interface."""
    args = parse_args()
    if args.command == "train":
        train_sdf_from_mesh(
            args.mesh_path,
            args.save_path,
            device=args.device,
            learning_rate=args.learning_rate,
            epoch_count=args.epochs,
            batch_size=args.batch_size,
            save_every_epochs=args.save_every,
            padding=args.padding,
            spacing=args.spacing,
            max_distance_points=args.max_distance_points,
        )


if __name__ == "__main__":
    main()
