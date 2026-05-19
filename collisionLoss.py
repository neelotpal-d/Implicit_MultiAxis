"""Collision and clearance losses for neural scalar fields.

The training scripts treat the gradient of a scalar field as a local build /
tool direction. This module samples a small "tool envelope" around that
direction and penalises scalar-field orderings that would imply
self-intersection, layer collision, or tool/model interference.

Public API:

- :class:`CollisionLoss` — additive (deposition) manufacturing.
- :class:`ToolProfile` and :data:`TOOL_PROFILES` — tool envelope presets
  selectable by name. The profile encodes the physical envelope; update
  the entry when the tool shape, nozzle/cutter radius, or required
  clearance changes.

The cone half-angle (``angle`` in :class:`CollisionLoss`) and the per-profile
distances/radii are in the same unit as the un-normalised mesh; the
``CollisionLoss.init_tool`` call divides by mesh scale to convert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sdfField import sdfModel

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DENOM_FLOOR: float = 1e-10
"""Floor added to vector norms before normalisation, to avoid div-by-zero."""

_SCALAR_COMP_RATIO_NEAR: float = 1e-4
"""Compensation ratio for the near cone loss; offsets the boundary slightly
to remove zero-margin degeneracy when sampled values exactly equal the
base scalar."""

_SCALAR_COMP_RATIO_FAR: float = 2e-4
"""Compensation ratio for the far cylinder losses (see _SCALAR_COMP_RATIO_NEAR)."""

_ERROR_SHARPNESS: float = 10.0
"""Multiplier on the violation before relu; controls how steeply the loss
ramps once a sample crosses the threshold."""

_FAR_DIST_JITTER_FRAC: float = 0.75
"""Fraction of the first-two-sample spacing used as random jitter on the
far-cylinder distances in ``collision_scalar_loss_far2``."""


# ---------------------------------------------------------------------------
# Tool envelope presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolProfile:
    """Tool envelope geometry consumed by :meth:`CollisionLoss.init_tool`.

    The values describe a three-layer envelope used by the collision checks:

    - ``dist_vals``: along-gradient distances probed inside the forward cone.
    - ``dist_array_far`` / ``radi_far``: distances and radius of the outer
      tangent-plane cylinder used by the far cone checks.
    - ``dist_array_in`` / ``radi_in``: distances and radius of the inner
      tangent-plane cylinder used by the inner cone checks.

    Distances and radii are expressed in mesh units; ``init_tool`` divides
    by the mesh scale to convert to the normalised coordinate system.
    """

    dist_vals: tuple[float, ...]
    dist_array_far: tuple[float, ...]
    radi_far: float
    dist_array_in: tuple[float, ...]
    radi_in: float


TOOL_PROFILES: dict[str, ToolProfile] = {
    "standard": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 11.0, 12.75),
        dist_array_far=(22.5, 33.75, 45.0),
        radi_far=25.0,
        dist_array_in=(7.5, 11.25, 33.75),
        radi_in=11.25,
    ),
    "dense": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 12.75),
        dist_array_far=(22.5, 33.75, 35.0, 40.0, 42.5, 45.0, 47.5, 50.0, 55.0, 60.0, 65.0, 70.0),
        radi_far=25.0,
        dist_array_in=(4.3, 7.5, 11.25, 33.75, 44.5, 49.5, 52.5, 57.5, 70.0, 80.0),
        radi_in=11.25,
    ),
    # Fertility / clip experiments use the "dense_uniform1" preset.
    # Note: the original code set radi_far=24 for fertility and 28 for clip;
    # the published value 24 is kept here. Override per-experiment by adding
    # a new entry rather than editing this one.
    "dense_uniform1": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 12.75),
        dist_array_far=(18.0, 23.0, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5),
        radi_far=24.0,
        dist_array_in=(4.5, 7.25, 11.25, 33.75, 44.5, 49.5, 52.5, 57.5),
        radi_in=11.25,
    ),
    "dense_uniform2": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 12.75),
        dist_array_far=(25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0),
        radi_far=25.0,
        dist_array_in=(7.5, 11.25, 33.75, 44.5, 49.5, 52.5, 57.5, 70.0, 80.0),
        radi_in=11.25,
    ),
    "dense1": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 12.75),
        dist_array_far=(22.5, 35.0, 40.0, 42.5, 70.0, 80.0, 90.0),
        radi_far=25.0,
        dist_array_in=(7.5, 11.25, 33.75, 44.5, 60.0),
        radi_in=11.25,
    ),
    "dense2": ToolProfile(
        dist_vals=(3.75, 7.5, 9.75, 12.75),
        dist_array_far=(33.75, 45.0, 50.0, 55.0, 60.0, 65.0, 85.0),
        radi_far=25.0,
        dist_array_in=(7.5, 52.5, 57.5, 70.0, 80.0),
        radi_in=11.25,
    ),
}


# ---------------------------------------------------------------------------
# Cone and circle samplers
# ---------------------------------------------------------------------------


def sample_directions_in_cone(directions: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Rotate canonical-frame samples to align with each input direction.

    Args:
        directions: (n, 3) unit-direction vectors.
        samples: (m, 3) canonical samples drawn from the +Z-aligned cone.

    Returns:
        (n, m, 3) tensor of sampled directions in the local frame of each
        input direction.
    """
    n = directions.shape[0]

    # Build a local frame for every input direction, then express the
    # canonical cone samples in that frame.
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=directions.device).expand(n, 3).clone()
    # If a direction is close to the chosen "arbitrary" vector, swap to a
    # different axis to avoid a degenerate cross product.
    close_to_x = torch.abs(directions[:, 0]) > 0.99
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=directions.device)

    u = torch.nn.functional.normalize(torch.cross(arbitrary_vector, directions, dim=-1), dim=1)
    v = torch.cross(directions, u, dim=-1)

    rotated_samples = (
        samples[:, 0:1] * u.unsqueeze(1)
        + samples[:, 1:2] * v.unsqueeze(1)
        + samples[:, 2:3] * directions.unsqueeze(1)
    )
    return rotated_samples  # (n, m, 3)


def get_cone_sample_direction_cosines3(angle: float, m: int, device: str = "cuda") -> torch.Tensor:
    """Sample ``m`` unit directions inside a cone of half-angle ``angle`` (deg).

    Layout: one axis sample, then ``m // 4`` stratified at ``theta / 2`` with
    uniformly spaced phi, then the remaining samples biased toward the cone
    boundary via cos^n sampling with deterministically spaced phi.

    Args:
        angle: Cone opening half-angle in degrees.
        m: Number of samples to draw.
        device: Torch device to place the samples on.

    Returns:
        ``(m, 3)`` tensor of direction cosines in the +Z-aligned canonical
        frame.
    """
    theta = torch.tensor(angle * torch.pi / 180)  # opening angle in radians

    # First sample: exactly along the cone axis.
    samples = [torch.tensor([[0.0, 0.0, 1.0]], device=device)]

    # Second group: m/4 samples at alpha = theta/2 with uniform phi.
    num_fixed_alpha = m // 4
    phi_fixed = torch.linspace(0.0, 2 * torch.pi, num_fixed_alpha, device=device)
    alpha_fixed = theta / 2
    cos_alpha_fixed = torch.cos(alpha_fixed)
    sin_alpha_fixed = torch.sin(alpha_fixed)

    samples.append(
        torch.stack(
            (
                sin_alpha_fixed * torch.cos(phi_fixed),
                sin_alpha_fixed * torch.sin(phi_fixed),
                cos_alpha_fixed * torch.ones_like(phi_fixed),
            ),
            dim=-1,
        )
    )

    # Third group: (3m/4 - 1) samples biased toward the cone boundary.
    # NOTE: the original code wrote ``(1 - 1) * torch.rand(...) ** n`` which
    # collapsed every sample in this group to ``cos(theta)``. We now bias
    # randomly between ``cos(theta)`` (boundary) and 1 (axis) via cos^n.
    num_remaining = m - (num_fixed_alpha + 1)
    phi_remaining = torch.linspace(0.0, 2 * torch.pi, num_remaining, device=device)
    n_bias = 5
    cos_alpha_remaining = torch.cos(theta) + (1 - torch.cos(theta)) * torch.rand(num_remaining, device=device) ** n_bias
    sin_alpha_remaining = torch.sqrt(1 - cos_alpha_remaining**2)

    samples.append(
        torch.stack(
            (
                sin_alpha_remaining * torch.cos(phi_remaining),
                sin_alpha_remaining * torch.sin(phi_remaining),
                cos_alpha_remaining,
            ),
            dim=-1,
        )
    )

    return torch.cat(samples, dim=0).to(device)


def sample_points_along_directions(
    sampled_directions: torch.Tensor, dist_vals: torch.Tensor, origins: torch.Tensor
) -> torch.Tensor:
    """Translate each direction sample by a set of distances and origins.

    Args:
        sampled_directions: ``(n, m, 3)`` direction unit vectors.
        dist_vals: ``(k,)`` distances along each direction.
        origins: ``(n, 3)`` origin points per cone.

    Returns:
        ``(n, m, k, 3)`` sampled points.
    """
    n = sampled_directions.shape[0]
    k = dist_vals.shape[0]

    dist_view = dist_vals.view(1, 1, k, 1)
    sampled_points = sampled_directions.unsqueeze(2) * dist_view  # (n, m, k, 3)
    return sampled_points + origins.view(n, 1, 1, 3)


def sample_tangent_circle(
    base_points: torch.Tensor,
    gradients: torch.Tensor,
    m: int,
    d: float | torch.Tensor,
    *,
    randomize_phase: bool = False,
) -> torch.Tensor:
    """Sample ``m`` points on a tangent-plane circle of radius ``d``.

    The gradient is treated as the local layer normal; the sampled circle
    approximates a cutter/nozzle footprint around the base point.

    Args:
        base_points: ``(n, 3)`` origin points.
        gradients: ``(n, 3)`` gradient (= layer normal) vectors at each point.
        m: Number of circle samples per point.
        d: Circle radius.
        randomize_phase: If True, jitter the starting angle so consecutive
            calls don't always test the same spokes. Used by the ``_far2``
            collision sampler to de-correlate the cylinder check across
            batches.

    Returns:
        ``(n, m, 3)`` tangent-circle samples.
    """
    n = base_points.shape[0]
    device = base_points.device

    normal = gradients / (gradients.norm(dim=-1, keepdim=True) + DENOM_FLOOR)

    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3).clone()
    close_to_x = torch.abs(normal[:, 0]) > 0.9
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=device)

    u = torch.cross(normal, arbitrary_vector, dim=-1)
    u = u / (u.norm(dim=-1, keepdim=True) + DENOM_FLOOR)
    v = torch.cross(normal, u, dim=-1)
    v = v / (v.norm(dim=-1, keepdim=True) + DENOM_FLOOR)

    if randomize_phase:
        phase = (2 * torch.pi / m) * float(np.random.rand())
        theta = torch.linspace(phase, 2 * torch.pi + phase, m, device=device).view(1, m, 1)
    else:
        theta = torch.linspace(0.0, 2 * torch.pi, m, device=device).view(1, m, 1)

    circle_offsets = d * (torch.cos(theta) * u.unsqueeze(1) + torch.sin(theta) * v.unsqueeze(1))
    return base_points.unsqueeze(1) + circle_offsets  # (n, m, 3)


def sample_along_gradient(
    tangent_samples: torch.Tensor, gradients: torch.Tensor, distances: torch.Tensor
) -> torch.Tensor:
    """Sweep tangent-plane footprint points along the local gradient direction.

    Args:
        tangent_samples: ``(n, m, 3)`` points sampled on the tangent plane.
        gradients: ``(n, 3)`` gradient direction at each base point.
        distances: ``(k,)`` distances along the gradient.

    Returns:
        ``(n, m, k, 3)`` swept points.
    """
    k = distances.shape[0]
    grad_unit = gradients / (gradients.norm(dim=-1, keepdim=True) + DENOM_FLOOR)
    displacement = grad_unit.unsqueeze(1).unsqueeze(2) * distances.view(1, 1, k, 1)  # (n, 1, k, 3)
    return tangent_samples.unsqueeze(2) + displacement  # (n, m, k, 3)


# ---------------------------------------------------------------------------
# CollisionLoss
# ---------------------------------------------------------------------------


class CollisionLoss:
    """Additive-style collision loss.

    Checks whether samples in the local forward tool envelope have scalar
    values that should already be "behind" the base point. Positive errors
    mean the learned scalar ordering would let layer/tool geometry cut into
    occupied or earlier material.

    Use :meth:`init_tool` to configure the physical envelope from a named
    preset in :data:`TOOL_PROFILES`. The preset must match the tool used
    during data collection; see :class:`ToolProfile` for the geometry it
    encodes.
    """

    def __init__(
        self,
        sample_num: int,
        angle: float,
        device: str = "cuda",
        distList: tuple[float, ...] | list[float] = (0.05, 0.1, 0.4, 0.8),
        model_load_path: str | None = None,
    ) -> None:
        # Cone directions are generated once and rotated to each gradient
        # during loss evaluation; distances and radii are configured by
        # init_tool.
        self.sampled_directions_seed = get_cone_sample_direction_cosines3(angle, sample_num, device=device)
        self.dist_vals = torch.tensor(list(distList), dtype=torch.float32, device=device)
        self.sdfModel: sdfModel | None = None
        if model_load_path:
            self.sdfModel = sdfModel(device=device, model_load_path=model_load_path)
        self.device = device
        self.dist_array_far: torch.Tensor | None = None
        self.radi_far: float | None = None
        self.dist_array_in: torch.Tensor | None = None
        self.radi_in: float | None = None

    def init_tool(self, profile_name: str, scale: float = 1.0) -> None:
        """Configure the tool envelope from a named preset.

        Args:
            profile_name: Key into :data:`TOOL_PROFILES`.
            scale: Mesh scale; distances and radii are divided by this
                value to convert from mesh units to the normalised
                coordinate system used by the field model.

        Raises:
            KeyError: If ``profile_name`` is not a known preset.
        """
        if profile_name not in TOOL_PROFILES:
            raise KeyError(
                f"Unknown tool profile {profile_name!r}; known profiles: {sorted(TOOL_PROFILES)!r}"
            )

        profile = TOOL_PROFILES[profile_name]
        s = float(scale)
        device = self.device

        self.dist_vals = torch.tensor(profile.dist_vals, dtype=torch.float32, device=device) / s
        self.dist_array_far = torch.tensor(profile.dist_array_far, dtype=torch.float32, device=device) / s
        self.radi_far = profile.radi_far / s
        self.dist_array_in = torch.tensor(profile.dist_array_in, dtype=torch.float32, device=device) / s
        self.radi_in = profile.radi_in / s

    # ----- internal helpers -------------------------------------------------

    def _combined_in_mask(self, sample_points: torch.Tensor, limVals) -> torch.Tensor:
        """Combine the axis-aligned domain mask with the optional SDF mask.

        ``limFun`` is only consulted when an SDF model is loaded; this lets
        the loss focus on samples inside the modelled part / valid domain.
        """
        in_mask = self.limModel(sample_points, limVals)
        if self.sdfModel is not None:
            in_mask_sdf = self.limFun(sample_points)
            assert in_mask.shape == in_mask_sdf.shape
            in_mask = in_mask * in_mask_sdf
        return in_mask

    @staticmethod
    def _scalar_compensation(scalars: torch.Tensor, ratio: float) -> torch.Tensor:
        """Tiny offset based on the scalar field range; avoids zero-margin
        degeneracy when sampled values are numerically equal to the base
        scalar."""
        return ratio * (torch.max(scalars) - torch.min(scalars))

    @staticmethod
    def _violation_loss(
        errors: torch.Tensor,
        in_mask: torch.Tensor,
        *,
        sharpness: float = _ERROR_SHARPNESS,
        squeeze_last: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hinge-loss reduction shared by the scalar collision losses.

        Returns ``(mean_squared_violation, error_mask)``.
        """
        error_samples = torch.relu(sharpness * errors)
        if squeeze_last:
            error_mask = (error_samples > 0)[..., 0] * in_mask
        else:
            error_mask = (error_samples > 0) * in_mask
        ms_error = torch.mean(error_samples[in_mask] * error_samples[in_mask])
        return ms_error, error_mask

    # ----- collision losses -------------------------------------------------

    def collision_scalar_loss(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        scalars: torch.Tensor,
        scalarField,
        limVals=(1.0, 1.0, 1.0),
    ) -> dict:
        """Cone-envelope loss based on scalar ordering along candidate tool paths."""
        grad_unit = grads / (grads.norm(dim=1, keepdim=True) + DENOM_FLOOR)

        sampled_directions = sample_directions_in_cone(grad_unit, self.sampled_directions_seed)
        sample_points = sample_points_along_directions(sampled_directions, self.dist_vals, points)
        scalar_at_samples = scalarField(sample_points)["scalars"]

        scalars_at_base = scalars.unsqueeze(1).unsqueeze(2)
        scalar_comp = self._scalar_compensation(scalars, _SCALAR_COMP_RATIO_NEAR)
        errors = scalars_at_base - scalar_at_samples + scalar_comp.detach()

        in_mask = self._combined_in_mask(sample_points, limVals)
        ms_error, error_mask = self._violation_loss(errors, in_mask, squeeze_last=True)

        return {
            "loss": ms_error,
            "directions": sampled_directions,
            "samples": sample_points,
            "mask": error_mask,
            "in_mask": in_mask,
        }

    def collision_scalar_loss_far(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        scalars: torch.Tensor,
        scalarField,
        limVals=(1.0, 1.0, 1.0),
        dist_array=(0.3, 0.45, 0.6),
        radi_: float = 0.3,
        n_angles: int = 10,
    ) -> dict:
        """Check a cylindrical envelope farther from the base point."""
        radi = self.radi_far if self.radi_far is not None else radi_
        sample_points_at_base = sample_tangent_circle(points, grads, n_angles, radi)

        if self.dist_array_far is not None:
            sample_distances = self.dist_array_far
        else:
            sample_distances = torch.tensor(dist_array, device=self.device)

        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances)
        scalar_at_samples = scalarField(sample_points)["scalars"]
        scalars_at_base = scalars.unsqueeze(1).unsqueeze(2)
        scalar_comp = self._scalar_compensation(scalars, _SCALAR_COMP_RATIO_FAR)
        errors = scalars_at_base - scalar_at_samples + scalar_comp.detach()

        in_mask = self._combined_in_mask(sample_points, limVals)
        ms_error, error_mask = self._violation_loss(errors, in_mask, squeeze_last=True)

        return {"loss": ms_error, "samples": sample_points, "mask": error_mask, "in_mask": in_mask}

    def collision_scalar_loss_far2(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        scalars: torch.Tensor,
        scalarField,
        limVals=(1.0, 1.0, 1.0),
        dist_array=(0.3, 0.45, 0.6),
        radi_: float = 0.3,
        n_angles: int = 10,
    ) -> dict:
        """Far envelope variant with randomised circle phase and jittered distances."""
        radi = self.radi_far if self.radi_far is not None else radi_
        sample_points_at_base = sample_tangent_circle(
            points, grads, n_angles, radi, randomize_phase=True
        )

        if self.dist_array_far is not None:
            jitter = (
                _FAR_DIST_JITTER_FRAC
                * (self.dist_array_far[1] - self.dist_array_far[0])
                * torch.rand(self.dist_array_far.shape, device=self.device)
            )
            sample_distances = self.dist_array_far + jitter
        else:
            sample_distances = torch.tensor(dist_array, device=self.device)

        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances)
        scalar_at_samples = scalarField(sample_points)["scalars"]
        scalars_at_base = scalars.unsqueeze(1).unsqueeze(2)
        scalar_comp = self._scalar_compensation(scalars, _SCALAR_COMP_RATIO_FAR)
        errors = scalars_at_base - scalar_at_samples + scalar_comp.detach()

        in_mask = self._combined_in_mask(sample_points, limVals)
        ms_error, error_mask = self._violation_loss(errors, in_mask, squeeze_last=True)

        return {"loss": ms_error, "samples": sample_points, "mask": error_mask, "in_mask": in_mask}

    def collision_scalar_loss_far_in(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        scalars: torch.Tensor,
        scalarField,
        limVals=(1.0, 1.0, 1.0),
        dist_array=(0.1, 0.15, 0.45),
        radi_: float = 0.15,
    ) -> dict:
        """Inner-radius envelope check for closer material/tool interference."""
        radi = self.radi_in if self.radi_in is not None else radi_
        sample_points_at_base = sample_tangent_circle(points, grads, 10, radi)

        if self.dist_array_in is not None:
            sample_distances = self.dist_array_in
        else:
            sample_distances = torch.tensor(dist_array, device=self.device)

        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances)
        scalar_at_samples = scalarField(sample_points)["scalars"]
        scalars_at_base = scalars.unsqueeze(1).unsqueeze(2)
        # NOTE: this method intentionally omits the scalar compensation term
        # used by the near / far losses. The inner envelope is asked to be
        # strictly above the base scalar, with no zero-margin slack.
        errors = scalars_at_base - scalar_at_samples

        in_mask = self._combined_in_mask(sample_points, limVals)
        ms_error, error_mask = self._violation_loss(errors, in_mask, squeeze_last=True)

        return {"loss": ms_error, "samples": sample_points, "mask": error_mask, "in_mask": in_mask}

    # ----- masks ------------------------------------------------------------

    def limModel(self, points: torch.Tensor, limVals) -> torch.Tensor:
        """Axis-aligned normalised-domain mask used before averaging errors."""
        x_mask = torch.abs(points[..., 0]) < limVals[0]
        y_mask = torch.abs(points[..., 1]) < limVals[1]
        z_mask = torch.abs(points[..., 2]) < limVals[2]
        return x_mask * y_mask * z_mask

    def limFun(self, points: torch.Tensor) -> torch.Tensor:
        """Optional learned-SDF mask; True where samples are inside the SDF model.

        Only meaningful when a SDF checkpoint was passed to ``__init__``.
        """
        if self.sdfModel is None:
            return torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        out_vals = self.sdfModel.predictOuts(points)
        in_mask = out_vals["scalars"] < 0
        assert in_mask.shape[-1] == 1
        return in_mask[..., 0]
