# Reproducing the results

This document describes what you can — and cannot — reproduce from the public repository today, and how.

**Short version:** Two figures (fertility support-free, T-bracket toolpath alignment) are reproducible from shipped checkpoints in minutes. Re-running those two trainings end-to-end takes roughly 4–12 h on one GPU and will **not** match exactly, because no random seeds are set. Everything else in the paper requires contacting the corresponding author.

## What ships with the repo

| Asset | Path | Purpose |
|---|---|---|
| Trained fertility layer field | `examples/checkpoints/parametersTest_batched_fertility_10_128_7_7.pt` | Final layer SIREN for the fertility support-free figure |
| Trained fertility SDF | `examples/checkpoints/fertilitySDF.pt` | Prerequisite for the collision loss when retraining fertility |
| Trained T-bracket bundle | `examples/outputs/toolpath_alignment_TshapeBracketNew/checkpoints/` | 23 saved epochs × {layer, toolpath, platform} from a completed 1100-epoch run |
| Initial T-bracket layer | `examples/checkpoints/parametersTest_batched_TshapeBracketNew_init.pt` | Warm-start for retraining the T-bracket pipeline |
| Configs | `examples/configs/support_free_config.json`, `examples/configs/toolpath_alignment_config.json` | Hyperparameters for the two shipped experiments |
| Stress data | `examples/inputs/stress_T_full.txt` | Used by the toolpath-alignment experiment |

What does **not** ship: configs / tool-geometry / SDFs / stress files for any other mesh in the paper. `examples/inputs/` contains 90+ files for clip, dyson, cupNew, freeCake, BodyS, sharpBar, etc., but only fertility and T-bracket have wired-up configs.

## Prerequisites

- Linux or macOS (Windows untested).
- Python 3.10+.
- CUDA-capable GPU for training. Inference / visualization works on CPU after editing `device` in the config, but the code hard-codes `device="cuda"` in several function defaults — expect to patch a few signatures if you have no GPU at all.
- A working display for PyVista (use `pyvista.start_xvfb()` on a headless server).

## Environment

Two paths, pick one. Both work on M1/M2/M3/M4 Macs (via MPS) and on Linux + CUDA.

### Pixi (recommended)

```bash
pixi install        # one-time, ~1 min
pixi run smoke      # validates the env: SIREN + 2nd-order autograd on the resolved device
pixi run support-free
pixi run toolpath-alignment
```

The `pyproject.toml` declares the dependencies under `[tool.pixi.*]`; `pixi install` produces a `pixi.lock` lockfile that pins exact versions for `osx-arm64` and `linux-64`.

### pip (fallback)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA (cu118, cu121, ...)
# on M-series macOS use plain `pip install torch` to get the MPS build
pip install -r requirements.txt
```

`requirements.txt` now uses minimum-version ranges; the `np.bool = np.bool_` monkey-patches in 4 source files make the code tolerant to both `numpy<2` and `numpy>=2.0`.

## Device selection

The shipped configs set `"device": "cuda"`. On non-CUDA hardware:

- Edit the config to `"device": "auto"` (picks cuda > mps > cpu) or `"device": "mps"` / `"device": "cpu"` explicitly.
- The pipeline resolves the device once at startup via `repro.resolve_device`, which raises a clear error if you ask for a device the host doesn't have rather than silently falling back.

## Tier 0 — render figures from shipped checkpoints

Use when you only want to see what the algorithm produced, not reproduce the training.

```bash
# fertility support-free figure
pixi run python examples/display_latest_checkpoint.py examples/configs/support_free_config.json

# T-bracket toolpath alignment figure
pixi run python examples/display_latest_checkpoint.py examples/configs/toolpath_alignment_config.json
```

The display script loads the latest checkpoint from the directories named in the config and opens a PyVista window. Each render takes seconds; the heavy work was already done.

**Determinism:** bit-exact — the weights are fixed in the shipped `.pt` files. **Verified on M1 Max via MPS** by `tests/test_smoke.py::test_shipped_fertility_checkpoint_loads_on_resolved_device`.

## Tier 1 — re-train the two shipped experiments

Use when you want to verify the training pipeline reaches a similar field.

```bash
# fertility (~4–8 h on a single A100-class GPU, 1100 epochs)
pixi run support-free

# T-bracket (~6–12 h, 1200 epochs)
pixi run toolpath-alignment
```

Outputs land in `examples/outputs/<output_dir>/`:

- `checkpoints/<prefix>_<component>_epoch_NNNNN.pt` every `save_every_epochs` epochs
- `losses/loss_*.txt` every `write_loss_every_epochs` epochs
- `losses/timing.txt`

**Determinism:** seed control is now wired in. Each pipeline calls `repro.set_global_seed(config.seed)` at startup, seeding `random`, `numpy`, `torch`, and `torch.cuda` (when present), plus disabling `cudnn.benchmark`. Two runs on the same machine with the same `seed` in the config now produce the same result. The default seed in `CommonTrainingConfig` is `42`.

Bit-exact reproduction of the *originally-shipped* checkpoints is still not possible — they predate seed control. But two `pixi run support-free` invocations on the same M1 Max with the same seed now produce identical checkpoints.

For bit-exact reproducibility on CUDA, additionally export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the run; cuBLAS atomics are otherwise non-deterministic.

## Tier 2 — reproduce any other figure from the paper

Not possible from the public repository alone. You will need from the corresponding author (`neelotpal.dutta@manchester.ac.uk`):

1. The **JSON config** for the additional mesh.
2. The **tool geometry parameters** for that mesh — currently embedded as comments in `collisionLoss.py:475` (`24 #24.00 change it to 24 for fertility 28 for clip`). The full mesh → radius mapping is not externalised.
3. The **SDF checkpoint** for that mesh, or recipes to train one — `python sdfField.py train --help`, ~30–60 min/mesh.
4. The **stress data file** if the experiment is stress-aligned. Only `stress_T_full.txt` is wired up; other `stress_*.txt` files in `examples/inputs/` are not referenced by any shipped config.
5. Which **`init_tool_*` profile** was used. The code defines six variants; only `init_tool_dense_uniform1` is called in the shipped pipelines.

## Tier 3 — bit-exact end-to-end reproduction

Not achievable today. Would require, at minimum:

- Seed control (see Tier 1).
- A locked dependency manifest (`pyproject.toml` with pinned versions, or `pixi.lock` / `conda-lock`).
- The exact GPU model used by the original authors documented (different CUDA architectures round floating-point differently).
- Per-mesh tool geometry externalised into configs rather than `collisionLoss.py` comments.

The originally-shipped checkpoints predate any seed control, so even after the above changes you can only achieve deterministic *new* runs — not bit-exact reproduction of the existing `.pt` files.

## Known issues you will hit

- **Numpy bool deprecation.** Patched in 4 files by `np.bool = np.bool_`. Works with `numpy<2`; will silently break with newer numpy.
- **PyVista version drift.** The display script uses APIs that may move between PyVista 0.42 → 0.44; pin it.
- **CUDA hard-coding.** Several function defaults assume `cuda`. CPU smoke tests require editing `device="cuda"` defaults in `sdfField.py` and `collisionLoss.py`.
- **Process-wide side effect.** `np.bool = np.bool_` mutates numpy globally on import — importing this code into a larger application affects unrelated code.
- **Silent NaN swallowing in collision losses** (`field_losses.py:84,96,108,120`). If a collision term goes NaN it is silently dropped from the loss while still polluting the logged metrics with NaN values. Watch for monotonically NaN-only entries in `losses/loss_collision.txt`.
- **One π-value inconsistency.** `collisionLoss.py:66`, `shared_geometry.py:12`, and `platform_losses.py:31` use the literal `3.1457` instead of `math.pi` for degree-to-radian conversion. This shifts the support-angle threshold by ~0.17°. It does not prevent training, but the shipped fields were optimized against a slightly-different boundary than a literal reading of the paper implies.

## Quick decision matrix

| Goal | Tier | Time |
|---|---|---|
| See what the algorithm produced for fertility / T-bracket | 0 | minutes |
| Verify the training pipeline reaches similar results | 1 | half a day per mesh |
| Reproduce a paper figure for another mesh | 2 (email author) | days, with iteration |
| Bit-exact regeneration for benchmarking | 3 (not possible today) | weeks of rework |
