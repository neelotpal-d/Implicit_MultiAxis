"""JSON-backed config object shared by example training pipelines."""

from dataclasses import dataclass
import json


@dataclass
class CommonTrainingConfig:
    """
    Wide config shared by structured training drivers.

    Some fields are intentionally unused by one workflow. Keeping them here
    makes experiment configs easier to compare and lets future setup code
    accept one config shape across different geometries.
    """

    model_name: str = ""
    device: str = "cuda"

    # Input files. Empty optional paths are allowed for workflows that do not
    # need stress data, toolpath checkpoints, platform checkpoints, or SDF data.
    surface_mesh_path: str = ""
    volume_mesh_path: str = ""
    stress_file_path: str = ""

    layer_checkpoint_path: str = ""
    toolpath_checkpoint_path: str = ""
    platform_checkpoint_path: str = ""
    sdf_checkpoint_path: str = ""

    load_toolpath_checkpoint: bool = False
    load_platform_checkpoint: bool = False

    # Mesh sampling and normalization controls.
    grid_padding: float = 0.0
    grid_spacing: float = 1.0
    sdf_keep_distance: float = 1.0
    max_input_points: int = 100000

    base_y_offset: float = 2.5
    boundary_base_offset: float = 2.5
    boundary_non_base_offset: float = -2.5

    # Batch sizes are expressed as target sizes because point counts vary by mesh.
    input_batch_target_size: int = 5000
    stress_batch_target_size: int = 5000
    boundary_batch_target_size: int = 5000
    base_batch_size: int = 20000

    layer_learning_rate: float = 5e-5
    toolpath_learning_rate: float = 5e-5
    platform_learning_rate: float = 5e-4
    layer_num_layers: int = 10
    layer_w0_initial: float = 7.0
    layer_w0: float = 7.0
    toolpath_num_layers: int = 15
    toolpath_w0_initial: float = 7.0
    toolpath_w0: float = 10.0
    scheduler_step_size: int = 150
    scheduler_gamma: float = 0.4
    epoch_count: int = 1000

    # Shared layer loss weights and curvature threshold.
    curvature_epsilons: tuple[float, float, float] = (1e-7, 2e-6, 1e-5)
    curvature_limit: float = 1.0 / 35.0
    layer_gradient_weight: float = 2e0
    layer_small_gradient_weight: float = 1e0
    layer_curvature_weight: float = 2e-1

    collision_sample_count: int = 20
    collision_angle_degrees: float = 61.0

    # Toolpath loss weights use the original staged update schedule.
    toolpath_gradient_norm_weight_initial: float = 2e0
    toolpath_curvature_weight_initial: float = 1e-3
    toolpath_gradient_norm_weight_max: float = 1.5e0
    toolpath_curvature_weight_max: float = 5e-1
    toolpath_gradient_norm_weight_update_every: int = 40
    toolpath_curvature_weight_update_every: int = 60
    platform_start_epoch: int = 1900

    # Output paths are folder-based; individual filenames are created by
    # training_outputs.py.
    output_dir: str = ""
    loss_log_dir: str = ""
    timing_log_name: str = "timing.txt"
    checkpoint_dir: str = ""
    checkpoint_prefix: str = ""

    save_every_epochs: int = 50
    write_loss_every_epochs: int = 20
    dry_run_batches: int | None = None
    show_progress: bool = True
    progress_every_epochs: int = 10
    progress_label: str = ""

    # Loss switches let experiments isolate one term without editing code.
    enable_losses: bool = True
    use_gradient_loss: bool = True
    use_curvature_loss: bool = True
    use_collision_loss: bool = True
    use_base_loss: bool = True
    use_boundary_support_loss: bool = True
    use_layer_loss: bool = True
    use_toolpath_loss: bool = True
    use_stress_loss: bool = True
    use_platform_loss: bool = True

    show_boundary_preview: bool = False
    show_stress_preview: bool = False


def loss_enabled(config, field_name):
    """Check the global loss switch and a specific loss switch."""
    return config.enable_losses and getattr(config, field_name)


def load_config(config_path):
    """Create a CommonTrainingConfig from a JSON file."""
    with open(config_path, "r") as file:
        values = json.load(file)

    config = CommonTrainingConfig()
    valid_fields = set(config.__dataclass_fields__.keys())
    unknown_fields = sorted(set(values.keys()) - valid_fields)
    if unknown_fields:
        raise ValueError(f"Unknown config fields in {config_path}: {unknown_fields}")

    for key, value in values.items():
        if key == "curvature_epsilons" and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)

    return config
