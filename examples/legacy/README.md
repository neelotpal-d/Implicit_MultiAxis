# Legacy scripts

This directory holds pre-refactor source files preserved for reference
when porting additional experiments to the modular pipeline.

- `tangentOptBatched_TshapeBracketNew_deepN.py` — the original
  monolithic T-bracket toolpath-alignment script, before the codebase
  was split into `experiment_loaders`, `field_losses`, `platform_losses`,
  `training_outputs`, and the example pipelines. Contains tool-geometry
  preset hard-codes (radii, sample distances) that may be useful when
  configuring `examples/configs/*.json` for new meshes.

These files are not exercised by the shipped pipelines or tests; do not
import from this directory. Treat it as a reference, not an API.
