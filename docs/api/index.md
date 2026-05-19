# API reference

Generated from the docstrings via `mkdocstrings-python`. Every public
function has a typed signature; every public class has a docstring
explaining what it represents.

## Layout

The codebase has nine top-level modules, divided into four categories:

<div class="grid cards" markdown>

- :material-cube-outline: **Geometry & math**

    - [`shared_geometry`](shared_geometry.md) — curvature formulas, support-angle loss
    - [`constants`](constants.md) — shared numerical tolerances

- :material-target: **Losses**

    - [`field_losses`](field_losses.md) — layer, base, boundary, toolpath, stress
    - [`collisionLoss`](collisionLoss.md) — tool-envelope collision proxy
    - [`platform_losses`](platform_losses.md) — platform half-space model

- :material-cog: **Training infrastructure**

    - [`training_dataclasses`](training_dataclasses.md) — config dataclass + JSON loader
    - [`experiment_loaders`](experiment_loaders.md) — mesh, stress, batching, model builders
    - [`repro`](repro.md) — device resolution + seed control

- :material-cube: **Models & data**

    - [`sdfField`](sdfField.md) — SDF SIREN + training loop + `SDFModel` wrapper

</div>

The vendored modified [`siren_pytorch`](https://github.com/lucidrains/siren-pytorch)
is documented in its own file; it adds explicit Hessian-row outputs (`HX2`,
`HY2`, `HZ2`) on top of the upstream forward pass.
