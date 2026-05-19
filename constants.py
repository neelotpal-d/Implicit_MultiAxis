"""Shared numerical tolerances and constants.

The original code inlined ``+ 1e-10`` (and a few outliers at ``+ 2e-10`` and
``+ 1e-8``) in 17 places across :mod:`shared_geometry`, :mod:`field_losses`,
:mod:`platform_losses`, :mod:`checkpoint_display`, and :mod:`collisionLoss`.
That made it impossible to tell intentional differences from copy-paste
drift. Promoting to a single named constant fixes both problems.

Without :mod:`compas` as a dependency, the value is a hand-rolled module
constant rather than a :class:`compas.tolerance.Tolerance`.
"""

from __future__ import annotations

DENOM_FLOOR: float = 1e-10
"""Floor added to vector norms before normalisation.

A unit-vector normalisation ``v / |v|`` is undefined at ``|v| = 0``. Adding
``DENOM_FLOOR`` to the denominator gives a smooth, well-defined behaviour:
unchanged unit vectors when ``|v| >> DENOM_FLOOR``, and a soft fall-off to
the zero vector as ``|v| -> 0``.

``1e-10`` is below float32 absolute precision for unit-scale vectors
(float32 has ~7 decimal digits) but is the de-facto pre-existing value used
throughout this codebase; standardising on it preserves training numerics
relative to the originally-shipped checkpoints.
"""
