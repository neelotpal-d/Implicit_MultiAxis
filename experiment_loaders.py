"""Data, loader, and model builders shared by the example pipelines."""

from itertools import cycle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared_utils import (
    build_sdf_masked_grid,
    load_mesh_pair,
    mesh_vertices_faces_bounds,
    normalization_tensors,
    normalize_points,
    shuffled_tensor_from_points,
)
from siren_pytorch import SirenNet


def load_geometry_bundle(config, boundary_mode="points"):
    """
    Load mesh-derived training data into one broad dictionary.

    The returned bundle intentionally contains more fields than every workflow
    needs. Training scripts can select the subset they use.
    """
    mesh, mesh_volume = load_mesh_pair(config.surface_mesh_path, config.volume_mesh_path)
    mesh_vertices, mesh_faces, mins, maxs = mesh_vertices_faces_bounds(mesh)
    # Build the layer sample set by keeping grid points near the input mesh.
    masked_grid = build_sdf_masked_grid(
        mesh_vertices,
        mesh_faces,
        mins,
        maxs,
        config.grid_padding,
        config.grid_spacing,
        config.sdf_keep_distance,
    )

    input_points = shuffled_tensor_from_points(masked_grid.points, config.max_input_points, config.device)
    base_points, boundary_points, boundary_normals = load_boundary_tensors(mesh, mins[1], config, boundary_mode)

    # All trainable points are normalized into the model coordinate system.
    mid_vals, range_vals = normalization_tensors(mins, maxs, config.device)
    input_points = normalize_points(input_points, mid_vals, range_vals)
    boundary_points = normalize_points(boundary_points, mid_vals, range_vals)
    base_points = normalize_points(base_points, mid_vals, range_vals)
    x_lim, y_lim, z_lim = torch.max(input_points, dim=0)[0]

    return {
        "mesh": mesh,
        "mesh_volume": mesh_volume,
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "mins": mins,
        "maxs": maxs,
        "masked_grid": masked_grid,
        "input_points": input_points,
        "base_points": base_points,
        "boundary_points": boundary_points,
        "boundary_normals": boundary_normals,
        "mid_vals": mid_vals,
        "range_vals": range_vals,
        "x_lim": x_lim,
        "y_lim": y_lim,
        "z_lim": z_lim,
    }


def load_boundary_tensors(mesh, y_min, config, boundary_mode="points"):
    """Load base/boundary tensors using either mesh point normals or face normals."""
    # Support-free examples use face centers/normals. Alignment examples use
    # mesh points/normals because platform loss acts on boundary samples.
    if boundary_mode == "faces":
        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        points_np = mesh.cell_centers().points
        normals_np = mesh["Normals"]
        split_offset = config.base_y_offset
        non_base_offset = config.base_y_offset
    else:
        mesh.compute_normals(cell_normals=False, inplace=True)
        points_np = mesh.points
        normals_np = mesh.point_data["Normals"]
        split_offset = config.boundary_base_offset
        non_base_offset = config.boundary_non_base_offset

    order = torch.randperm(points_np.shape[0]).cpu().numpy()
    points_np = points_np[order]
    normals_np = normals_np[order]

    y_points = points_np[:, 1]
    base_mask = y_points < (y_min + split_offset)
    non_base_mask = y_points >= (y_min + non_base_offset)

    base_points = torch.tensor(
        points_np[base_mask],
        dtype=torch.float32,
        device=config.device,
        requires_grad=True,
    )
    boundary_points = torch.tensor(
        points_np[non_base_mask],
        dtype=torch.float32,
        device=config.device,
        requires_grad=True,
    )
    boundary_normals = torch.tensor(
        normals_np[non_base_mask],
        dtype=torch.float32,
        device=config.device,
    )

    return base_points, boundary_points, boundary_normals


def load_stress_bundle(config, mid_vals, range_vals, input_count=None):
    """Load optional stress tensors and normalized/downsampled stress points."""
    if not config.stress_file_path:
        return {
            "stress_points": None,
            "stress_dirs": None,
            "stress_mags": None,
            "stress_weights": None,
        }

    stress_rows = np.loadtxt(config.stress_file_path)
    if stress_rows.ndim != 2 or stress_rows.shape[1] != 9:
        raise ValueError("Stress file must contain 9 columns per row.")

    stress_points = torch.tensor(stress_rows[:, 0:3], dtype=torch.float32, requires_grad=True, device=config.device)
    stress_dirs = torch.tensor(stress_rows[:, 3:6], dtype=torch.float32, device=config.device)
    stress_mags = torch.tensor(stress_rows[:, 6:9], dtype=torch.float32, device=config.device)

    stress_points = normalize_points(stress_points, mid_vals, range_vals)
    if input_count is not None:
        stress_points, stress_dirs, stress_mags = downsample_stress_points(
            stress_points,
            stress_dirs,
            stress_mags,
            input_count,
            config.device,
        )

    return {
        "stress_points": stress_points,
        "stress_dirs": stress_dirs,
        "stress_mags": stress_mags,
        "stress_weights": compute_stress_weights(stress_mags),
    }


def farthest_point_sampling_indices(points, num_samples, device):
    """Pick spread-out samples so stress points cover the part more evenly."""
    num_samples = int(num_samples)
    centroids = torch.zeros(num_samples, dtype=torch.long, device=device)
    distance = torch.ones(points.shape[0], dtype=torch.float32, device=device) * 1e10
    farthest = torch.randint(0, points.shape[0], (1,), dtype=torch.long, device=device)

    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance)

    return centroids


def downsample_stress_points(stress_points, stress_dirs, stress_mags, input_count, device):
    indices = farthest_point_sampling_indices(stress_points, int(input_count), device)
    stress_points = stress_points[indices].detach()
    stress_points.requires_grad = True
    stress_dirs = stress_dirs[indices].detach()
    stress_mags = stress_mags[indices]
    if device == "cuda":
        torch.cuda.empty_cache()
    return stress_points, stress_dirs, stress_mags


def compute_stress_weights(stress_mags):
    """Create stress weights and zero out compression samples."""
    av_max_stress = torch.mean(abs(stress_mags[:, 0]))
    stress_mag_ratios = abs(stress_mags[:, 0]) / av_max_stress
    m_temp = stress_mag_ratios / 1.0
    stress_mag_ratio_wts = 0.1 + (1 - 0.1) * (m_temp / (1 + m_temp))
    stress_mag_ratio_wts = stress_mag_ratio_wts.unsqueeze(1)
    stress_mag_ratio_wts = torch.ones_like(stress_mag_ratio_wts)

    compression_mask = stress_mags[:, 0] < 0
    stress_mag_ratio_wts[compression_mask.flatten()] = 0.0
    return stress_mag_ratio_wts


def make_input_stress_loaders(config, input_points, stress_points=None, stress_dirs=None, stress_weights=None):
    """Create loaders for layer points plus optional stress points."""
    input_batches = int(input_points.shape[0] / config.input_batch_target_size) + 1
    input_batch_size = int(input_points.shape[0] / input_batches) + 1
    input_loader = DataLoader(input_points, batch_size=input_batch_size, shuffle=True)

    if stress_points is None:
        return input_loader, None

    stress_batches = int(stress_points.shape[0] / config.stress_batch_target_size) + 1
    stress_batch_size = int(stress_points.shape[0] / stress_batches) + 1
    stress_dataset = TensorDataset(stress_points, stress_dirs, stress_weights)
    stress_loader = DataLoader(stress_dataset, batch_size=stress_batch_size, shuffle=True)
    return input_loader, stress_loader


def make_toolpath_alignment_loaders(config, input_points, stress_points, stress_dirs, stress_weights, boundary_points, boundary_normals):
    """Create input, stress, and boundary loaders for alignment workflows."""
    input_loader, stress_loader = make_input_stress_loaders(
        config,
        input_points,
        stress_points,
        stress_dirs,
        stress_weights,
    )

    boundary_batches = int(boundary_points.shape[0] / config.boundary_batch_target_size) + 1
    boundary_batch_size = int(boundary_points.shape[0] / boundary_batches) + 1
    boundary_dataset = TensorDataset(boundary_points, boundary_normals)
    boundary_loader = DataLoader(boundary_dataset, batch_size=boundary_batch_size, shuffle=True)

    print(len(input_loader))
    print(len(stress_loader))
    print(len(boundary_loader))

    # Match the longest essential loader so each iteration has all sample types.
    if len(input_loader) > len(stress_loader):
        stress_loader = cycle(stress_loader)
    elif len(stress_loader) > len(input_loader):
        input_loader = cycle(input_loader)

    boundary_loader = cycle(boundary_loader)

    return {
        "input_points": input_loader,
        "stress_points": stress_loader,
        "boundary_points": boundary_loader,
    }


def make_support_free_loaders(config, input_points, base_points, boundary_points, boundary_normals):
    """Create input, base, and boundary loaders for support-constrained workflows."""
    input_batches = int(input_points.shape[0] / config.input_batch_target_size) + 1
    boundary_batches = int(boundary_points.shape[0] / config.boundary_batch_target_size) + 1

    input_batch_size = int(input_points.shape[0] / input_batches) + 1
    boundary_batch_size = int(boundary_points.shape[0] / boundary_batches) + 1

    input_loader = DataLoader(input_points, batch_size=input_batch_size, shuffle=True)
    base_loader = DataLoader(base_points, batch_size=config.base_batch_size, shuffle=True)
    boundary_dataset = TensorDataset(boundary_points, boundary_normals)
    boundary_loader = DataLoader(boundary_dataset, batch_size=boundary_batch_size, shuffle=True)

    print(len(input_loader))
    print(len(boundary_loader))
    print(len(base_loader))

    # Base points are always cycled because they are an auxiliary constraint.
    if len(input_loader) > len(boundary_loader):
        boundary_loader = cycle(boundary_loader)
        base_loader = cycle(base_loader)
    elif len(boundary_loader) > len(input_loader):
        input_loader = cycle(input_loader)
        base_loader = cycle(base_loader)
    else:
        base_loader = cycle(base_loader)

    return {
        "input_points": input_loader,
        "base_points": base_loader,
        "boundary_points": boundary_loader,
    }


def build_scalar_field(config, checkpoint_path, num_layers, w0_initial, w0):
    """Create a SIREN scalar field and optionally load a checkpoint."""
    scalar_field = SirenNet(3, 128, 1, num_layers, w0_initial=w0_initial, w0=w0).to(config.device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        scalar_field.load_state_dict(checkpoint["model_state_dict"])
    return scalar_field


def build_field_bundle(config, include_secondary=True, platform_factory=None, platform_scale=1.0):
    """
    Build a broad model bundle with optional second field and platform params.

    Scripts can use only the entries they need; for example a layer-only run can
    ignore `scalar_field2` and `platform`.
    """
    # The first scalar field represents layers; the optional second field
    # represents toolpath direction within those layers.
    scalar_field = build_scalar_field(
        config,
        config.layer_checkpoint_path,
        num_layers=config.layer_num_layers,
        w0_initial=config.layer_w0_initial,
        w0=config.layer_w0,
    )

    scalar_field2 = None
    if include_secondary:
        scalar_field2 = build_scalar_field(
            config,
            "",
            num_layers=config.toolpath_num_layers,
            w0_initial=config.toolpath_w0_initial,
            w0=config.toolpath_w0,
        )
        if config.load_toolpath_checkpoint and config.toolpath_checkpoint_path:
            checkpoint2 = torch.load(config.toolpath_checkpoint_path)
            scalar_field2.load_state_dict(checkpoint2["model_state_dict"])

    platform = None
    if platform_factory is not None:
        platform = platform_factory(device=config.device, scale=platform_scale).to(config.device)
        if config.load_platform_checkpoint and config.platform_checkpoint_path:
            checkpoint3 = torch.load(config.platform_checkpoint_path)
            platform.load_state_dict(checkpoint3["model_state_dict"])

    return {
        "scalar_field": scalar_field,
        "scalar_field2": scalar_field2,
        "platform": platform,
    }
