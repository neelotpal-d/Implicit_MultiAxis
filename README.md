
## Modules

- `training_dataclasses.py`: common JSON-backed config and loss switches.
- `experiment_loaders.py`: mesh, stress, model, and `DataLoader` construction helpers.
- `field_losses.py`: reusable scalar-field, support, collision, toolpath, and stress loss functions.
- `shared_geometry.py`: curvature, support, geodesic-curvature, and inside-mask utilities.
- `shared_utils.py`: mesh loading, SDF-filtered grid generation, normalization, and tensor helpers.
- `sdfField.py`: reusable SIREN SDF model, checkpoint helpers, SDF training losses, and mesh sample preparation.
- `platform_losses.py`: reusable platform model and platform-position/clearance loss.
- `collisionLoss.py`: collision loss implementation. The tool initialization presets in this file are for the current tool geometry and must be updated when tool shape, size, or clearance assumptions change.

## Setup

Recommended via pixi (one-shot install, lockfile-pinned, picks up MPS on Apple Silicon and CUDA on Linux):

```bash
pixi install
pixi run smoke          # validates the env on the resolved device
pixi run support-free   # runs the fertility pipeline end-to-end
```

Or via pip (install PyTorch separately so it matches your CUDA / MPS / CPU setup):

```bash
pip install torch       # M-series Macs get MPS automatically; for CUDA see https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The pipeline configs default to `"device": "cuda"`. On Apple Silicon set it to `"auto"` (which picks `cuda > mps > cpu`) or to `"mps"` explicitly. `repro.resolve_device` raises a clear error if you ask for a device the host doesn't have rather than silently falling back.

The structured scripts expect the local `siren_pytorch` folder to remain in this repository.

See `reproduce.md` for the full reproducibility recipe (tiered by ambition: render shipped checkpoints, retrain shipped configs, reproduce other paper figures, bit-exact regeneration).

## Examples

Run the support-free workflow (Fertility Model):

```bash
python examples/support_free_pipeline.py examples/configs/support_free_config.json
```

Run the toolpath-alignment workflow (T-bracket Model):

```bash
python examples/toolpath_alignment_pipeline.py examples/configs/toolpath_alignment_config.json
```


## Loss Switches

Losses can be enabled or disabled in each JSON config:

```json
"enable_losses": true,
"use_gradient_loss": true,
"use_curvature_loss": true,
"use_collision_loss": true,
"use_base_loss": true,
"use_boundary_support_loss": true,
"use_layer_loss": true,
"use_toolpath_loss": true,
"use_stress_loss": true,
"use_platform_loss": true
```

Set `enable_losses` to `false` to disable all optional loss terms, or set an individual `use_*_loss` field to `false` for targeted experiments.

The common layer loss also has configurable weights:

```json
"layer_gradient_weight": 2.0,
"layer_small_gradient_weight": 1.0,
"layer_curvature_weight": 0.2
```

The toolpath loss has two staged weights:

```json
"toolpath_gradient_norm_weight_initial": 2.0,
"toolpath_curvature_weight_initial": 0.001
```

`toolpath_gradient_norm_weight_*` controls the projected-gradient norm term for the toolpath field. `toolpath_curvature_weight_*` controls the geodesic-curvature penalty for toolpaths.

SIREN architecture frequency settings are also in config:

```json
"layer_num_layers": 10,
"layer_w0_initial": 7.0,
"layer_w0": 7.0,
"toolpath_num_layers": 15,
"toolpath_w0_initial": 7.0,
"toolpath_w0": 10.0
```

## Outputs

Training outputs are written under each config's `output_dir`:

- `losses/loss_total.txt`
- `losses/loss_layer.txt`
- `losses/loss_toolpath.txt`
- `losses/loss_stress.txt`
- `losses/loss_cross.txt`
- `losses/loss_collision.txt`
- `losses/loss_base.txt`
- `losses/loss_boundary.txt`
- `losses/loss_platform.txt`
- `losses/timing.txt`
- `checkpoints/<checkpoint_prefix>_<checkpoint_item>_epoch_<epoch>.pt`

Checkpoints are saved every `save_every_epochs`. Loss files are written every `write_loss_every_epochs`. Set `show_progress` to `true` to show a `tqdm` progress bar when available, or plain-text progress every `progress_every_epochs` otherwise.

## Displaying Checkpoints

`checkpoint_display.py` contains reusable PyVista display helpers. By default, each helper loads the latest checkpoint from the config's checkpoint folder. Pass `epoch=` to load a specific saved epoch, or pass explicit checkpoint paths when comparing files manually.
`

To display the latest checkpoint in the output folder named by a config:

```bash
python examples/display_latest_checkpoint.py examples/configs/support_free_config.json
python examples/display_latest_checkpoint.py examples/configs/toolpath_alignment_config.json
```

## SDF 

Train an SDF from a mesh:

```bash
python sdfField.py train examples/inputs/fertility.obj --save-path examples/checkpoints/fertilitySDF.pt --epochs 150 --spacing 0.2 --device cuda
```

Use `--max-distance-points` when the SDF grid is too large for memory.


## Citation

If you use this code, please cite the accompanying paper:

> Dutta, N., Zhang, T., Liu, T., Chen, Y., & Wang, C.C.L. (2026).
> *Implicit Neural Field-Based Process Planning for Multi-Axis Manufacturing.*
> Computer-Aided Design (accepted).

A `CITATION.cff` is provided for GitHub's auto-citation widget and for
Zotero / Citation File Format-aware tools.

Project page: <https://neelotpal-d.github.io/Implicit_MultiAxis/>

## License

MIT. See `LICENSE`. The vendored `siren_pytorch/` directory is a modified
copy of [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch),
also MIT licensed; its original copyright is reproduced in `LICENSE`.

## Contact

If you have any questions, drop us a mail at [neelotpal.dutta@manchester.ac.uk](mailto:neelotpal.dutta@manchester.ac.uk)