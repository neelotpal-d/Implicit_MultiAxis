"""Example training pipeline for support-free layer-field optimization."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow this example script to import shared modules from the repository root
# when launched directly as `python examples/support_free_pipeline.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pyvista as pv
import torch
from torch.optim import lr_scheduler

from collisionLoss import CollisionLoss
from experiment_loaders import build_field_bundle, load_geometry_bundle, make_support_free_loaders
from field_losses import (
    add_base_loss,
    add_boundary_support_loss,
    add_collision_losses,
    compute_layer_loss,
)
from repro import resolve_device, set_global_seed
from training_dataclasses import CommonTrainingConfig, load_config, loss_enabled
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


@dataclass(frozen=True)
class SupportFreePreparedData:
    """Geometry tensors and normalization values needed by the support-free run."""

    mesh: pv.PolyData
    mesh_volume: pv.DataSet
    input_points: torch.Tensor
    base_points: torch.Tensor
    boundary_points: torch.Tensor
    boundary_normals: torch.Tensor
    mid_vals: torch.Tensor
    range_vals: torch.Tensor
    x_lim: torch.Tensor
    y_lim: torch.Tensor
    z_lim: torch.Tensor


def prepare_data(config: CommonTrainingConfig) -> SupportFreePreparedData:
    """Load mesh samples and boundary/base points for support-free training."""
    bundle = load_geometry_bundle(config, boundary_mode="faces")
    return SupportFreePreparedData(
        mesh=bundle["mesh"],
        mesh_volume=bundle["mesh_volume"],
        input_points=bundle["input_points"],
        base_points=bundle["base_points"],
        boundary_points=bundle["boundary_points"],
        boundary_normals=bundle["boundary_normals"],
        mid_vals=bundle["mid_vals"],
        range_vals=bundle["range_vals"],
        x_lim=bundle["x_lim"],
        y_lim=bundle["y_lim"],
        z_lim=bundle["z_lim"],
    )


def build_scalar_field(config: CommonTrainingConfig):
    """Build the single layer scalar field used by this workflow."""
    return build_field_bundle(config, include_secondary=False)["scalar_field"]


def build_collision_loss(data: SupportFreePreparedData, config: CommonTrainingConfig):
    """Create the collision loss with the current tool/sample settings."""
    collision_loss = CollisionLoss(
        config.collision_sample_count,
        config.collision_angle_degrees,
        device=config.device,
        model_load_path=config.sdf_checkpoint_path,
    )
    collision_loss.init_tool_dense_uniform1(scale=data.range_vals[0])
    return collision_loss


def run_training(config: CommonTrainingConfig):
    """Run the full support-free training loop."""
    data = prepare_data(config)
    loaders = make_support_free_loaders(
        config,
        data.input_points,
        data.base_points,
        data.boundary_points,
        data.boundary_normals,
    )
    scalar_field = build_scalar_field(config)

    optimizer = torch.optim.Adam(list(scalar_field.parameters()), lr=config.layer_learning_rate)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    collision_loss = build_collision_loss(data, config) if loss_enabled(config, "use_collision_loss") else None
    loss_records = init_loss_records()
    progress = epoch_progress(config)
    start = time.time()

    for epoch in progress:
        # The loaders are cycled as needed by experiment_loaders, so one zip
        # gives each iteration layer samples, base samples, and boundary normals.
        zipped_set = zip(loaders["input_points"], loaders["base_points"], loaders["boundary_points"])
        iter_count = 0
        last_loss = None
        batch_losses = None

        for batch_input_points, batch_base_points, (batch_bound_points, batch_bound_normals) in zipped_set:
            iter_count += 1

            # Layer loss is shared by all workflows; support-free then adds
            # collision, base-orientation, and boundary support constraints.
            loss, layer_record, out, grads = compute_layer_loss(batch_input_points, scalar_field, data, config)
            if batch_losses is None:
                batch_losses = init_batch_losses(loss)

            loss, collision_record = add_collision_losses(
                loss, batch_input_points, out, grads, scalar_field, collision_loss, data, config, epoch
            )
            loss, base_record = add_base_loss(loss, scalar_field, batch_base_points, config, epoch)
            loss, boundary_record = add_boundary_support_loss(
                loss, scalar_field, batch_bound_points, batch_bound_normals, config, epoch
            )

            loss.backward()
            last_loss = loss
            if torch.isnan(loss):
                break

            optimizer.step()
            optimizer.zero_grad()
            batch_losses["total"] += loss
            batch_losses["layer"] += layer_record
            batch_losses["collision"] += collision_record
            batch_losses["base"] += base_record
            batch_losses["boundary"] += boundary_record

            if config.dry_run_batches is not None and iter_count >= config.dry_run_batches:
                break

        if last_loss is not None and torch.isnan(last_loss):
            print("nan loss detected")
            break

        if epoch % config.save_every_epochs == 0:
            # All checkpoints and timing logs use the common examples/outputs layout.
            save_checkpoint_bundle(config, epoch, {"scalar_field": scalar_field})
            write_timing_log(config, epoch, start)

        scheduler.step()

        if iter_count > 0 and batch_losses is not None:
            append_loss_records(loss_records, batch_losses, iter_count)
            update_progress(progress, {"loss": loss_records["total"][-1]})

        if epoch % config.write_loss_every_epochs == 0:
            write_loss_logs(config, loss_records)

        if config.dry_run_batches is not None:
            break

    write_loss_logs(config, loss_records)

    if config.show_boundary_preview:
        show_boundary_preview(data)

    return scalar_field, data, loss_records


def show_boundary_preview(data: SupportFreePreparedData):
    """Preview the selected boundary points in original mesh coordinates."""
    picked_points = data.boundary_points
    new_points = picked_points * data.range_vals + data.mid_vals
    new_mesh = pv.PolyData(new_points.to("cpu").detach().numpy())

    plotter = pv.Plotter()
    plotter.add_mesh(new_mesh, color="red")
    plotter.add_mesh(data.mesh, style="wireframe")
    plotter.show(cpos="xy")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/configs/support_free_config.json"
    config = load_config(config_path)
    # Resolve 'auto'/'cuda'/'mps'/'cpu' once at startup and write it back into
    # the config so every downstream module sees a concrete device name.
    config.device = str(resolve_device(config.device))
    set_global_seed(config.seed)
    print(f"device={config.device}  seed={config.seed}")
    run_training(config)


if __name__ == "__main__":
    main()
