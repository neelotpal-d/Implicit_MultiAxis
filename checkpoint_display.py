"""Utilities for visualizing saved layer/toolpath checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from constants import DENOM_FLOOR
from experiment_loaders import build_scalar_field
from platform_losses import PlatformModel
from shared_utils import load_mesh_pair, mesh_vertices_faces_bounds, normalization_tensors
from training_dataclasses import load_config


def latest_checkpoint(checkpoint_dir, pattern):
    """Return the newest checkpoint matching a glob pattern."""
    paths = sorted(Path(checkpoint_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched {Path(checkpoint_dir) / pattern}")
    return str(paths[-1])


def checkpoint_exists(config, item_name):
    """Return whether a saved checkpoint exists for one trainable item."""
    prefix = config.checkpoint_prefix or (config.model_name or "model")
    checkpoint_dir = checkpoint_dir_from_config(config)
    return any(checkpoint_dir.glob(f"{prefix}_{item_name}_epoch_*.pt"))


def checkpoint_dir_from_config(config):
    """Return the configured checkpoint folder."""
    return Path(config.checkpoint_dir or Path(config.output_dir) / "checkpoints")


def checkpoint_path_for_epoch(config, item_name, epoch):
    """Build the common checkpoint filename for one trainable item."""
    prefix = config.checkpoint_prefix or (config.model_name or "model")
    checkpoint_dir = checkpoint_dir_from_config(config)
    return str(checkpoint_dir / f"{prefix}_{item_name}_epoch_{epoch:05d}.pt")


def resolve_checkpoint_path(config, item_name, explicit_path=None, epoch=None, fallback_path=""):
    """Find a checkpoint by explicit path, epoch, latest saved output, then fallback."""
    if explicit_path:
        return str(explicit_path)

    if epoch is not None:
        return checkpoint_path_for_epoch(config, item_name, epoch)

    prefix = config.checkpoint_prefix or (config.model_name or "model")
    checkpoint_dir = checkpoint_dir_from_config(config)
    try:
        return latest_checkpoint(checkpoint_dir, f"{prefix}_{item_name}_epoch_*.pt")
    except FileNotFoundError:
        if fallback_path:
            return str(fallback_path)
        raise


def load_checkpoint_state(model, checkpoint_path, device):
    """Load a saved `model_state_dict` into a model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_display_geometry(config):
    """Load mesh geometry plus normalization tensors used by the trained field."""
    mesh, mesh_volume = load_mesh_pair(config.surface_mesh_path, config.volume_mesh_path)
    _, _, mins, maxs = mesh_vertices_faces_bounds(mesh)
    mid_vals, range_vals = normalization_tensors(mins, maxs, config.device)
    return mesh, mesh_volume, mid_vals, range_vals


def load_layer_field(config, checkpoint_path):
    """Build and load the layer scalar field."""
    field = build_scalar_field(
        config,
        "",
        num_layers=config.layer_num_layers,
        w0_initial=config.layer_w0_initial,
        w0=config.layer_w0,
    )
    return load_checkpoint_state(field, checkpoint_path, config.device)


def load_toolpath_field(config, checkpoint_path):
    """Build and load the toolpath scalar field."""
    field = build_scalar_field(
        config,
        "",
        num_layers=config.toolpath_num_layers,
        w0_initial=config.toolpath_w0_initial,
        w0=config.toolpath_w0,
    )
    return load_checkpoint_state(field, checkpoint_path, config.device)


def load_platform_model(config, checkpoint_path, range_vals):
    """Build and load the platform model."""
    platform = PlatformModel(device=config.device, scale=range_vals[0]).to(config.device)
    return load_checkpoint_state(platform, checkpoint_path, config.device)


def normalize_points(points, mid_vals, range_vals, device):
    """Convert physical points into trained field coordinates."""
    points = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
    points = (points - mid_vals) / range_vals
    points = points.detach()
    points.requires_grad = True
    return points


def evaluate_field_on_points(field, points):
    """Evaluate scalar field values on a tensor of normalized points."""
    with torch.enable_grad():
        out = field(points)
    return out["scalars"].detach().to("cpu").numpy().flatten()


def add_layer_scalars(mesh_volume, layer_field, mid_vals, range_vals, device, scalar_name="layer_scalar"):
    """Attach layer-field scalar values to mesh-volume points."""
    normalized_points = normalize_points(mesh_volume.points, mid_vals, range_vals, device)
    scalars = evaluate_field_on_points(layer_field, normalized_points)
    mesh_volume = mesh_volume.copy()
    mesh_volume.point_data[scalar_name] = scalars
    return mesh_volume, scalars


def layer_values_from_scalars(scalars, layer_count=15, padding=1e-4):
    """Create evenly spaced layer contour values inside the scalar range."""
    min_scalar = float(np.min(scalars)) + padding
    max_scalar = float(np.max(scalars)) - padding
    if max_scalar <= min_scalar:
        raise ValueError("Layer scalar range is too small to contour.")
    return np.linspace(min_scalar, max_scalar, layer_count)


def build_layer_contours(mesh_volume, layer_values, scalar_name="layer_scalar"):
    """Extract layer surfaces from a scalar-valued mesh volume."""
    return mesh_volume.contour(layer_values, scalars=scalar_name)


def subdivide_layer_surface(layer):
    """Apply one Loop subdivision pass to a triangular layer surface."""
    import igl

    if layer.n_points < 3 or layer.faces.size == 0:
        return layer

    vertices = layer.points
    faces = layer.faces.reshape(-1, 4)[:, 1:]
    subdivided_vertices, subdivided_faces = igl.loop(vertices, faces)
    faces_pv = np.hstack((np.full((subdivided_faces.shape[0], 1), 3), subdivided_faces))
    return pv.PolyData(subdivided_vertices, faces_pv.astype(np.int64))


def add_toolpath_curves_to_plotter(
    plotter,
    layer_contours,
    toolpath_field,
    mid_vals,
    range_vals,
    device,
    curve_count=30,
    tube_radius=0.2,
    color="black",
    global_toolpath_range=None,
):
    """Evaluate the toolpath field on layer surfaces and draw contour curves."""
    if layer_contours.n_points < 3 or curve_count < 2:
        return

    normalized_points = normalize_points(layer_contours.points, mid_vals, range_vals, device)
    toolpath_scalars = evaluate_field_on_points(toolpath_field, normalized_points)
    scalar_range = float(toolpath_scalars.max() - toolpath_scalars.min())
    if scalar_range <= 1e-8:
        return

    if global_toolpath_range and global_toolpath_range > 1e-8:
        curve_count = max(2, int(curve_count * scalar_range / global_toolpath_range))

    layer_contours = layer_contours.copy()
    layer_contours.point_data["toolpath_scalar"] = toolpath_scalars
    padding = min(0.003, scalar_range * 0.25)
    start_value = float(toolpath_scalars.min() + padding)
    end_value = float(toolpath_scalars.max() - padding)
    if end_value <= start_value:
        return

    values = np.linspace(start_value, end_value, curve_count)
    curves = layer_contours.contour(values, scalars="toolpath_scalar")
    if curves.n_points < 3:
        return

    plotter.add_mesh(curves.tube(radius=tube_radius, n_sides=20), color=color)


def add_platform_plane_to_plotter(plotter, platform, mid_vals, range_vals, size=250):
    """Draw the learned platform plane in physical coordinates."""
    platform_pos = platform.platform_base.detach() * range_vals + mid_vals
    platform_dir = platform.platform_dir.detach()
    platform_dir = platform_dir / (torch.norm(platform_dir, dim=1, keepdim=True) + DENOM_FLOOR)

    center = platform_pos.squeeze(0).to("cpu").numpy()
    direction = platform_dir.squeeze(0).to("cpu").numpy()
    plane = pv.Plane(center=center, direction=direction, i_size=size, j_size=size)
    plotter.add_mesh(plane, color="lightgray", opacity=0.35)


def display_support_free_checkpoint(config_path, checkpoint_path=None, epoch=None, layer_count=15, cpos="xy"):
    """Display layer contours for a support-free checkpoint."""
    config = load_config(config_path)
    checkpoint_path = resolve_checkpoint_path(
        config,
        "scalar_field",
        explicit_path=checkpoint_path,
        epoch=epoch,
        fallback_path=config.layer_checkpoint_path,
    )
    mesh, mesh_volume, mid_vals, range_vals = load_display_geometry(config)
    layer_field = load_layer_field(config, checkpoint_path)

    mesh_volume, scalars = add_layer_scalars(mesh_volume, layer_field, mid_vals, range_vals, config.device)
    layer_values = layer_values_from_scalars(scalars, layer_count=layer_count)
    layers = build_layer_contours(mesh_volume, layer_values)

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(layers, cmap="rainbow")
    plotter.add_mesh(mesh, style="wireframe", color="gray", opacity=0.25)
    plotter.show(cpos=cpos)
    return layers


def display_toolpath_alignment_checkpoint(
    config_path,
    layer_checkpoint_path=None,
    toolpath_checkpoint_path=None,
    platform_checkpoint_path=None,
    epoch=None,
    layer_count=15,
    curve_count=30,
    tube_radius=0.2,
    subdivide_layers=False,
    show_platform=True,
    cpos="zy",
):
    """Display layer contours and toolpath curves for an alignment checkpoint."""
    config = load_config(config_path)
    layer_checkpoint_path = resolve_checkpoint_path(
        config,
        "scalar_field",
        explicit_path=layer_checkpoint_path,
        epoch=epoch,
        fallback_path=config.layer_checkpoint_path,
    )
    toolpath_checkpoint_path = resolve_checkpoint_path(
        config,
        "scalar_field2",
        explicit_path=toolpath_checkpoint_path,
        epoch=epoch,
        fallback_path=config.toolpath_checkpoint_path,
    )

    mesh, mesh_volume, mid_vals, range_vals = load_display_geometry(config)
    layer_field = load_layer_field(config, layer_checkpoint_path)
    toolpath_field = load_toolpath_field(config, toolpath_checkpoint_path)

    mesh_volume, scalars = add_layer_scalars(mesh_volume, layer_field, mid_vals, range_vals, config.device)
    layer_values = layer_values_from_scalars(scalars, layer_count=layer_count)
    all_layers = build_layer_contours(mesh_volume, layer_values)
    all_layer_points = normalize_points(all_layers.points, mid_vals, range_vals, config.device)
    all_toolpath_scalars = evaluate_field_on_points(toolpath_field, all_layer_points)
    global_toolpath_range = float(all_toolpath_scalars.max() - all_toolpath_scalars.min())

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing("ssaa")
    plotter.renderer.SetUseDepthPeeling(True)

    for value in layer_values:
        layer = build_layer_contours(mesh_volume, [value])
        if layer.n_points < 3:
            continue
        if subdivide_layers:
            layer = subdivide_layer_surface(layer)
        plotter.add_mesh(layer, cmap="rainbow", opacity=0.65)
        add_toolpath_curves_to_plotter(
            plotter,
            layer,
            toolpath_field,
            mid_vals,
            range_vals,
            config.device,
            curve_count=curve_count,
            tube_radius=tube_radius,
            global_toolpath_range=global_toolpath_range,
        )

    if show_platform:
        platform_checkpoint_path = resolve_checkpoint_path(
            config,
            "platform",
            explicit_path=platform_checkpoint_path,
            epoch=epoch,
            fallback_path=config.platform_checkpoint_path,
        )
        platform = load_platform_model(config, platform_checkpoint_path, range_vals)
        add_platform_plane_to_plotter(plotter, platform, mid_vals, range_vals)

    plotter.add_mesh(mesh, style="wireframe", color="gray", opacity=0.2)
    plotter.show(cpos=cpos)
    return plotter


def display_latest_checkpoint(
    config_path,
    mode="auto",
    layer_count=15,
    curve_count=30,
    tube_radius=0.2,
    subdivide_layers=False,
    show_platform=True,
    cpos=None,
):
    """Display the latest checkpoint in the output folder named by a config."""
    config = load_config(config_path)

    if mode == "auto":
        mode = "toolpath" if checkpoint_exists(config, "scalar_field2") else "layer"

    if mode == "layer":
        return display_support_free_checkpoint(
            config_path,
            layer_count=layer_count,
            cpos=cpos or "xy",
        )

    if mode == "toolpath":
        return display_toolpath_alignment_checkpoint(
            config_path,
            layer_count=layer_count,
            curve_count=curve_count,
            tube_radius=tube_radius,
            subdivide_layers=subdivide_layers,
            show_platform=show_platform,
            cpos=cpos or "zy",
        )

    raise ValueError(f"Unknown display mode: {mode}")
