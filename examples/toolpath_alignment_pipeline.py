"""Example training pipeline for layer and toolpath alignment."""

from dataclasses import dataclass
from pathlib import Path
import sys
import time

# Allow this example script to import shared modules from the repository root
# when launched directly as `python examples/toolpath_alignment_pipeline.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from experiment_loaders import (
    build_field_bundle,
    load_geometry_bundle,
    load_stress_bundle,
    make_toolpath_alignment_loaders,
)
from field_losses import compute_layer_loss, compute_stress_losses, compute_toolpath_loss
from platform_losses import PlatformModel, add_platform_loss
from training_dataclasses import CommonTrainingConfig, load_config
from training_outputs import (
    append_loss_records,
    epoch_progress,
    init_batch_losses,
    init_loss_records,
    save_checkpoint_bundle,
    update_progress,
    write_loss_logs,
    write_timing_log,
)


np.bool = np.bool_


@dataclass
class ToolpathAlignmentPreparedData:
    """Geometry, stress samples, and normalization tensors for alignment runs."""

    mesh: pv.PolyData
    mesh_volume: pv.DataSet
    input_points: torch.Tensor
    stress_points: torch.Tensor
    stress_dirs: torch.Tensor
    stress_mags: torch.Tensor
    stress_weights: torch.Tensor
    boundary_points: torch.Tensor
    boundary_normals: torch.Tensor
    base_points: torch.Tensor
    mid_vals: torch.Tensor
    range_vals: torch.Tensor
    x_lim: torch.Tensor
    y_lim: torch.Tensor
    z_lim: torch.Tensor


@dataclass
class ToolpathAlignmentModels:
    """The layer field, toolpath field, and platform model trained together."""

    scalar_field: nn.Module
    scalar_field2: nn.Module
    platform: nn.Module


def prepare_data(config: CommonTrainingConfig) -> ToolpathAlignmentPreparedData:
    """Load mesh samples plus stress samples used for toolpath alignment."""
    geometry = load_geometry_bundle(config, boundary_mode="points")
    stress = load_stress_bundle(
        config,
        geometry["mid_vals"],
        geometry["range_vals"],
        input_count=geometry["input_points"].shape[0],
    )

    print("---")
    print(geometry["input_points"].shape)
    print(stress["stress_points"].shape)
    print("---")

    return ToolpathAlignmentPreparedData(
        mesh=geometry["mesh"],
        mesh_volume=geometry["mesh_volume"],
        input_points=geometry["input_points"],
        stress_points=stress["stress_points"],
        stress_dirs=stress["stress_dirs"],
        stress_mags=stress["stress_mags"],
        stress_weights=stress["stress_weights"],
        boundary_points=geometry["boundary_points"],
        boundary_normals=geometry["boundary_normals"],
        base_points=geometry["base_points"],
        mid_vals=geometry["mid_vals"],
        range_vals=geometry["range_vals"],
        x_lim=geometry["x_lim"],
        y_lim=geometry["y_lim"],
        z_lim=geometry["z_lim"],
    )


def build_models(config: CommonTrainingConfig, data: ToolpathAlignmentPreparedData) -> ToolpathAlignmentModels:
    """Build the layer field, toolpath field, and optional platform model."""
    bundle = build_field_bundle(config, platform_factory=PlatformModel, platform_scale=data.range_vals[0])
    return ToolpathAlignmentModels(
        scalar_field=bundle["scalar_field"],
        scalar_field2=bundle["scalar_field2"],
        platform=bundle["platform"],
    )


def build_optimizers(config: CommonTrainingConfig, models: ToolpathAlignmentModels):
    """Create separate optimizers because fields and platform use different rates."""
    optimizer1 = torch.optim.Adam(list(models.scalar_field.parameters()), lr=config.layer_learning_rate)
    optimizer2 = torch.optim.Adam(list(models.scalar_field2.parameters()), lr=config.toolpath_learning_rate)
    optimizer3 = torch.optim.Adam(list(models.platform.parameters()), lr=config.platform_learning_rate)

    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    return optimizer1, optimizer2, optimizer3, scheduler1, scheduler2


def update_toolpath_weights(epoch: int, config: CommonTrainingConfig, gradient_norm_weight: float, curvature_weight: float):
    """Increase toolpath loss weights using the original experiment schedule."""
    if (
        (epoch + 1) % config.toolpath_gradient_norm_weight_update_every == 0
        and gradient_norm_weight < config.toolpath_gradient_norm_weight_max
    ):
        gradient_norm_weight *= 2
        print("updating toolpath gradient-norm weight")

    if (epoch + 1) % config.toolpath_curvature_weight_update_every == 0:
        if curvature_weight < config.toolpath_curvature_weight_max:
            curvature_weight *= 2
            print("updating toolpath curvature weight")
        else:
            curvature_weight = 1e0

    return gradient_norm_weight, curvature_weight


def run_training(config: CommonTrainingConfig):
    """Run the full toolpath-alignment training loop."""
    data = prepare_data(config)
    loaders = make_toolpath_alignment_loaders(
        config,
        data.input_points,
        data.stress_points,
        data.stress_dirs,
        data.stress_weights,
        data.boundary_points,
        data.boundary_normals,
    )
    models = build_models(config, data)
    optimizer1, optimizer2, optimizer3, scheduler1, scheduler2 = build_optimizers(config, models)

    loss_records = init_loss_records()

    toolpath_gradient_norm_weight = config.toolpath_gradient_norm_weight_initial
    toolpath_curvature_weight = config.toolpath_curvature_weight_initial
    start = time.time()
    progress = epoch_progress(config)

    for epoch in progress:
        # Each iteration pairs layer samples, stress samples, and boundary
        # samples. Loader cycling is handled in experiment_loaders.
        zipped_set = zip(loaders["input_points"], loaders["stress_points"], loaders["boundary_points"])
        iter_count = 0
        last_loss = None
        batch_losses = None

        full_scalars = models.scalar_field(data.input_points)["scalars"].detach()
        batch_max = torch.max(full_scalars).detach()
        batch_min = torch.min(full_scalars).detach()
        batch_range = batch_max - batch_min
        batch_range_lim = 0.5 * batch_range

        for batch_input_points, (batch_s_points, batch_s_dirs, batch_s_wts), (batch_boundary_points, batch_boundary_normals) in zipped_set:
            iter_count += 1

            # The shared layer loss trains the first scalar field. Toolpath,
            # stress, and platform losses then constrain the second field and
            # the platform model around that layer field.
            loss, layer_record, primary_out, _ = compute_layer_loss(
                batch_input_points,
                models.scalar_field,
                data,
                config,
                master_switch="use_layer_loss",
            )
            if batch_losses is None:
                batch_losses = init_batch_losses(loss)

            tp_loss, tp_record, _ = compute_toolpath_loss(
                models,
                batch_input_points,
                primary_out,
                batch_min,
                batch_max,
                batch_range_lim,
                data,
                config,
                toolpath_gradient_norm_weight,
                toolpath_curvature_weight,
            )
            loss += tp_loss

            stress_loss_out = compute_stress_losses(
                models,
                batch_s_points,
                batch_s_dirs,
                batch_s_wts,
                batch_min,
                batch_max,
                batch_range_lim,
                epoch,
                config,
            )
            stress_loss, layer_stress_record, cross_record, tp_stress_record = stress_loss_out
            loss += stress_loss

            loss, platform_record = add_platform_loss(
                loss,
                models.platform,
                models.scalar_field,
                batch_boundary_points,
                batch_boundary_normals,
                epoch,
                config,
            )

            loss.backward()
            last_loss = loss
            if torch.isnan(loss):
                break

            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            if epoch > config.platform_start_epoch:
                optimizer3.step()
                optimizer3.zero_grad()

            batch_losses["total"] += loss
            batch_losses["layer"] += layer_record + layer_stress_record
            batch_losses["toolpath"] += tp_record
            batch_losses["stress"] += stress_loss
            batch_losses["cross"] += cross_record
            batch_losses["platform"] += platform_record

            if config.dry_run_batches is not None and iter_count >= config.dry_run_batches:
                break

        if last_loss is not None and torch.isnan(last_loss):
            print("nan loss detected")
            break

        toolpath_gradient_norm_weight, toolpath_curvature_weight = update_toolpath_weights(
            epoch,
            config,
            toolpath_gradient_norm_weight,
            toolpath_curvature_weight,
        )

        if epoch % config.save_every_epochs == 0:
            # Checkpoints for all trainable components use the same naming scheme.
            save_checkpoint_bundle(
                config,
                epoch,
                {
                    "scalar_field": models.scalar_field,
                    "scalar_field2": models.scalar_field2,
                    "platform": models.platform,
                },
            )
            write_timing_log(config, epoch, start)

        if iter_count > 0 and batch_losses is not None:
            append_loss_records(loss_records, batch_losses, iter_count)
            update_progress(progress, {"loss": loss_records["total"][-1]})

        scheduler1.step()
        scheduler2.step()

        if epoch % config.write_loss_every_epochs == 0:
            write_loss_logs(config, loss_records)

        if config.dry_run_batches is not None:
            break

    write_loss_logs(config, loss_records)

    if config.show_stress_preview:
        show_stress_preview(data)

    return models, data, loss_records


def show_stress_preview(data: ToolpathAlignmentPreparedData):
    """Preview sampled stress points in original mesh coordinates."""
    new_points = data.stress_points * data.range_vals + data.mid_vals
    new_mesh = pv.PolyData(new_points.to("cpu").detach().numpy())

    plotter = pv.Plotter()
    plotter.add_mesh(new_mesh, color="red")
    plotter.add_mesh(data.mesh, style="wireframe")
    plotter.show()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/configs/toolpath_alignment_config.json"
    config = load_config(config_path)
    run_training(config)


if __name__ == "__main__":
    main()

