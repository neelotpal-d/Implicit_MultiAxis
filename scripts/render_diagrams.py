"""Render conceptual diagrams of the algorithm into ``docs/figures/``.

Six figures:

1. ``cone_sampler.png`` — the canonical-frame cone direction sampler,
   showing the stratification (axis + half-angle ring + boundary-biased
   tail) used by :func:`collisionLoss.get_cone_sample_direction_cosines3`.
2. ``tangent_circle.png`` — the tangent-plane circle sampler used for
   the far / inner cylinder collision checks.
3. ``tool_envelope.png`` — a 2D cross-section of the three-layer tool
   envelope (cone + outer cylinder + inner cylinder) defined by a
   :class:`collisionLoss.ToolProfile`.
4. ``support_angle.png`` — the support-angle loss intuition: cosine of
   the angle between the outward normal and the build direction.
5. ``layer_field.png`` — implicit layers as level sets of the scalar
   field, in 2D for clarity.
6. ``loss_composition.png`` — block diagram of how layer / collision /
   base / boundary / toolpath / stress / platform losses combine.

Run with: ``pixi run python scripts/render_diagrams.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Cone sampler
# ---------------------------------------------------------------------------


def render_cone_sampler() -> None:
    """3D view of the cone direction sampler.

    Axis sample (red dot at +Z), half-angle ring (orange), boundary-biased
    samples (blue, denser near the rim by cos^n weighting).
    """
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="3d")

    angle_deg = 60.0
    theta = np.radians(angle_deg)

    # Wireframe of the cone surface.
    h = np.linspace(0, 1.0, 24)
    phi = np.linspace(0, 2 * np.pi, 64)
    H, P = np.meshgrid(h, phi)
    radius = H * np.tan(theta)
    Xs = radius * np.cos(P)
    Ys = radius * np.sin(P)
    Zs = H
    ax.plot_wireframe(Xs, Ys, Zs, color="lightgray", linewidth=0.4, alpha=0.6)

    # Group 1: axis sample.
    ax.scatter([0], [0], [1.0], color="crimson", s=90, label="axis sample (1)", zorder=5)

    # Group 2: half-angle ring (m/4 samples).
    m = 24
    n_ring = m // 4
    phi_ring = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    alpha_ring = theta / 2
    cos_a = np.cos(alpha_ring)
    sin_a = np.sin(alpha_ring)
    ax.scatter(
        sin_a * np.cos(phi_ring),
        sin_a * np.sin(phi_ring),
        np.full_like(phi_ring, cos_a),
        color="darkorange",
        s=55,
        label=f"half-angle ring ({n_ring})",
        zorder=4,
    )

    # Group 3: boundary-biased tail (cos^n).
    rng = np.random.default_rng(0)
    n_tail = m - n_ring - 1
    phi_tail = np.linspace(0, 2 * np.pi, n_tail, endpoint=False)
    n_bias = 5
    cos_tail = np.cos(theta) + (1 - np.cos(theta)) * rng.uniform(size=n_tail) ** n_bias
    sin_tail = np.sqrt(1 - cos_tail**2)
    ax.scatter(
        sin_tail * np.cos(phi_tail),
        sin_tail * np.sin(phi_tail),
        cos_tail,
        color="steelblue",
        s=45,
        label=f"boundary-biased ({n_tail})",
        zorder=3,
    )

    # Cone axis arrow.
    ax.quiver(0, 0, 0, 0, 0, 1.05, color="black", linewidth=1.2, arrow_length_ratio=0.06)
    ax.text(0, 0, 1.15, "build / tool direction", ha="center", fontsize=9)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (cone axis)")
    ax.set_title(f"Cone direction sampler\nhalf-angle = {angle_deg}°, stratified")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.view_init(elev=22, azim=-55)
    _save(fig, "cone_sampler.png")


# ---------------------------------------------------------------------------
# 2. Tangent circle
# ---------------------------------------------------------------------------


def render_tangent_circle() -> None:
    """3D view of the tangent-plane circle sampler.

    Origin point, gradient (= layer normal), tangent plane (light gray),
    and the m equally-spaced circle samples used to probe the cylinder
    radius of the tool envelope.
    """
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")

    origin = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.2, 0.3, 1.0])
    normal /= np.linalg.norm(normal)

    # Build an orthonormal frame {u, v, normal}.
    arbitrary = np.array([1.0, 0.0, 0.0])
    u = np.cross(arbitrary, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Tangent plane (mesh) — semi-transparent.
    g = np.linspace(-0.6, 0.6, 6)
    Gu, Gv = np.meshgrid(g, g)
    plane = origin[:, None, None] + u[:, None, None] * Gu + v[:, None, None] * Gv
    ax.plot_surface(plane[0], plane[1], plane[2], color="lightgray", alpha=0.35, edgecolor="none")

    # Circle samples.
    m = 12
    radius = 0.5
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pts = origin[:, None] + radius * (np.cos(angles) * u[:, None] + np.sin(angles) * v[:, None])
    ax.scatter(pts[0], pts[1], pts[2], color="steelblue", s=55, label=f"{m} circle samples", zorder=5)
    # Connect them lightly to read as a circle.
    ax.plot(
        np.append(pts[0], pts[0, 0]),
        np.append(pts[1], pts[1, 0]),
        np.append(pts[2], pts[2, 0]),
        color="steelblue",
        linewidth=0.8,
        alpha=0.6,
    )

    # Gradient / normal arrow.
    ax.quiver(*origin, *normal * 0.9, color="black", linewidth=1.6, arrow_length_ratio=0.08)
    ax.text(
        *(origin + normal * 1.0),
        "∇f  (layer normal /\n  tool direction)",
        fontsize=9,
    )
    # Origin marker.
    ax.scatter(*origin, color="crimson", s=80, zorder=6)
    ax.text(0.02, 0.02, -0.05, "base point", fontsize=9)

    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.3, 1.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Tangent-plane circle sampler\n(used by the cylinder collision check)")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=15, azim=-60)
    _save(fig, "tangent_circle.png")


# ---------------------------------------------------------------------------
# 3. Tool envelope cross-section
# ---------------------------------------------------------------------------


def render_tool_envelope() -> None:
    """2D cross-section of the three-layer tool envelope.

    Reads the ``dense_uniform1`` profile so the figure stays in sync if
    the table is edited.
    """
    from collisionLoss import TOOL_PROFILES

    profile = TOOL_PROFILES["dense_uniform1"]
    fig, ax = plt.subplots(figsize=(6.5, 7.5))

    # Cone (near zone).
    cone_angle = np.radians(30.0)  # half-angle for the diagram
    z_cone = np.linspace(0, max(profile.dist_vals), 50)
    r_cone = z_cone * np.tan(cone_angle)
    ax.fill_betweenx(z_cone, -r_cone, r_cone, color="crimson", alpha=0.15, label="near cone")
    for d in profile.dist_vals:
        # Plot cone sample distances along the axis.
        ax.axhline(d, color="crimson", linewidth=0.5, alpha=0.4)
    ax.scatter(
        np.zeros(len(profile.dist_vals)),
        profile.dist_vals,
        color="crimson",
        s=30,
        zorder=4,
        label="cone samples",
    )

    # Inner cylinder.
    ax.fill_betweenx(
        profile.dist_array_in,
        -profile.radi_in,
        profile.radi_in,
        color="darkorange",
        alpha=0.15,
        label="inner cylinder",
    )
    for d in profile.dist_array_in:
        ax.scatter(
            [-profile.radi_in, profile.radi_in],
            [d, d],
            color="darkorange",
            s=20,
            zorder=4,
        )

    # Far cylinder (outer).
    ax.fill_betweenx(
        profile.dist_array_far,
        -profile.radi_far,
        profile.radi_far,
        color="steelblue",
        alpha=0.12,
        label="far cylinder",
    )
    for d in profile.dist_array_far:
        ax.scatter(
            [-profile.radi_far, profile.radi_far],
            [d, d],
            color="steelblue",
            s=20,
            zorder=3,
        )

    # Base point at origin.
    ax.scatter([0], [0], color="black", s=80, zorder=10)
    ax.annotate(
        "base point\n(layer normal points up)",
        xy=(0, 0),
        xytext=(8, -8),
        fontsize=9,
        ha="left",
    )
    ax.arrow(0, 0, 0, 5, color="black", head_width=0.8, length_includes_head=True)
    ax.text(1.2, 4, "∇f", fontsize=10)

    ax.set_xlim([-profile.radi_far - 4, profile.radi_far + 4])
    ax.set_ylim([-2, max(profile.dist_array_far) + 5])
    ax.set_xlabel("radial offset (mesh units)")
    ax.set_ylabel("along-gradient distance (mesh units)")
    ax.set_aspect("equal")
    ax.set_title("Tool envelope: 'dense_uniform1' profile\n(cross-section through the gradient axis)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    _save(fig, "tool_envelope.png")


# ---------------------------------------------------------------------------
# 4. Support-angle loss
# ---------------------------------------------------------------------------


def render_support_angle() -> None:
    """Plot the support-angle hinge loss as a function of angle.

    Shows how the relu-of-(cos-threshold) construction creates a
    one-sided penalty whose strength scales with the violation angle.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    angles_deg = np.linspace(0, 180, 400)
    threshold_deg = 132.0
    sharpness = 25.0

    # support_error in shared_geometry.support_loss.
    cos_thr = np.cos(np.radians(threshold_deg))
    dot_prod = np.cos(np.radians(angles_deg))  # angle between normal and grad
    support_error = -dot_prod + cos_thr
    hinge = np.maximum(0.0, sharpness * support_error)
    loss = hinge**2

    ax.fill_between(angles_deg, 0, loss, color="crimson", alpha=0.18)
    ax.plot(angles_deg, loss, color="crimson", linewidth=2.0, label="hinge² loss")
    ax.axvline(threshold_deg, color="black", linestyle="--", linewidth=1.0, label=f"threshold = {threshold_deg}°")
    ax.set_xlabel("angle between outward normal and build direction (deg)")
    ax.set_ylabel("per-point loss")
    ax.set_xlim([0, 180])
    ax.set_title(
        f"Support-angle loss: relu(sharpness · (-cos θ + cos {threshold_deg:.0f}°))²\n(sharpness = {sharpness})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    _save(fig, "support_angle.png")


# ---------------------------------------------------------------------------
# 5. Layer field (2D level sets)
# ---------------------------------------------------------------------------


def render_layer_field() -> None:
    """Implicit layers as level sets of the scalar field.

    Synthetic 2D field that bends near the centre, with level lines drawn
    at evenly spaced scalar values. Conveys why curvature of layers
    must be bounded (tool steerability).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    grid = np.linspace(-1.5, 1.5, 200)
    xv, yv = np.meshgrid(grid, grid)
    # A bent layer field with mild curvature near the centre.
    field = yv + 0.45 * np.exp(-3 * xv**2) * np.cos(1.6 * yv)
    contours = ax.contour(xv, yv, field, levels=15, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    # Gradient field at sparse points.
    sg = np.linspace(-1.3, 1.3, 8)
    xs, ys = np.meshgrid(sg, sg)
    # Numerical gradient of `field` evaluated on a sparse grid.
    fx = -2.7 * xs * np.exp(-3 * xs**2) * np.cos(1.6 * ys)
    fy = 1 - 1.6 * 0.45 * np.exp(-3 * xs**2) * np.sin(1.6 * ys)
    mag = np.sqrt(fx**2 + fy**2)
    ax.quiver(xs, ys, fx / mag, fy / mag, color="steelblue", scale=22, width=0.003, alpha=0.6)

    ax.set_xlabel("x")
    ax.set_ylabel("y (build axis)")
    ax.set_title("Implicit layers as scalar level sets\n(arrows: ∇f, the local tool direction)")
    ax.set_aspect("equal")
    _save(fig, "layer_field.png")


# ---------------------------------------------------------------------------
# 6. Loss composition (block diagram)
# ---------------------------------------------------------------------------


def render_loss_composition() -> None:
    """Block diagram of how the loss terms compose into total loss."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 6.5])
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(xy1, xy2):
        ax.annotate(
            "",
            xy=xy2,
            xytext=xy1,
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9},
        )

    # Inputs.
    block(0.2, 5.4, 2.2, 0.7, "batch_input_points", "lightyellow")
    block(0.2, 4.4, 2.2, 0.7, "batch_base_points", "lightyellow")
    block(0.2, 3.4, 2.2, 0.7, "batch_boundary_points", "lightyellow")
    block(0.2, 2.4, 2.2, 0.7, "batch_stress_points", "lightyellow")
    block(0.2, 1.4, 2.2, 0.7, "boundary_points (PLT)", "lightyellow")

    # SIREN fields.
    block(3.2, 5.0, 2.2, 1.0, "layer scalar_field\n(SIREN, 10 layers)", "lavender")
    block(3.2, 2.6, 2.2, 1.0, "toolpath scalar_field2\n(SIREN, 15 layers)", "lavender")
    block(3.2, 1.0, 2.2, 1.0, "PlatformModel\n(base + dir params)", "lavender")

    # Losses.
    block(6.3, 5.7, 2.0, 0.6, "compute_layer_loss", "mistyrose")
    block(6.3, 5.0, 2.0, 0.6, "add_collision_losses\n(near/far/far2/inner)", "mistyrose")
    block(6.3, 4.3, 2.0, 0.6, "add_base_loss", "mistyrose")
    block(6.3, 3.6, 2.0, 0.6, "add_boundary_support_loss", "mistyrose")
    block(6.3, 2.9, 2.0, 0.6, "compute_toolpath_loss", "mistyrose")
    block(6.3, 2.2, 2.0, 0.6, "compute_stress_losses", "mistyrose")
    block(6.3, 1.5, 2.0, 0.6, "add_platform_loss", "mistyrose")

    # Output.
    block(9.0, 3.4, 1.8, 1.0, "total loss\n(scalar)", "honeydew")

    # Arrows from inputs to fields.
    arrow((2.4, 5.75), (3.2, 5.5))
    arrow((2.4, 4.75), (3.2, 5.5))  # base also goes to layer field
    arrow((2.4, 3.75), (3.2, 5.3))  # boundary -> layer field
    arrow((2.4, 2.75), (3.2, 5.3))  # stress -> layer field
    arrow((2.4, 2.75), (3.2, 3.1))  # stress -> toolpath
    arrow((2.4, 1.75), (3.2, 1.5))  # boundary (PLT) -> platform

    # Fields to losses.
    arrow((5.4, 5.5), (6.3, 6.0))
    arrow((5.4, 5.5), (6.3, 5.3))
    arrow((5.4, 5.5), (6.3, 4.6))
    arrow((5.4, 5.5), (6.3, 3.9))
    arrow((5.4, 3.1), (6.3, 3.2))
    arrow((5.4, 3.1), (6.3, 2.5))
    arrow((5.4, 1.5), (6.3, 1.8))

    # Losses to total.
    for ly in (6.0, 5.3, 4.6, 3.9, 3.2, 2.5, 1.8):
        arrow((8.3, ly), (9.0, 3.9))

    ax.set_title("Loss composition graph (pipelines call into field_losses)", fontsize=11)
    _save(fig, "loss_composition.png")


# ---------------------------------------------------------------------------
# 7. SIREN architecture
# ---------------------------------------------------------------------------


def render_siren_architecture() -> None:
    """Block diagram of the SIREN network used for both fields."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 3.6])
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9})

    # Input
    block(0.2, 1.6, 1.2, 0.7, "x ∈ ℝ³", "lightyellow")
    # Hidden layers
    xs = [1.8, 3.4, 5.0, 6.6]
    for x_pos in xs:
        block(x_pos, 1.6, 1.2, 0.7, "Linear\n+ sin(w₀·)", "lavender")
    block(8.2, 1.6, 1.4, 0.7, "Linear\n(linear out)", "lavender")
    block(10.0, 1.6, 0.8, 0.7, "f(x)", "honeydew")

    # Hessian branch
    block(8.2, 0.4, 1.4, 0.7, "autograd\n∂f/∂x, ∂²f/∂x²", "mistyrose")
    block(10.0, 0.4, 0.8, 0.7, "∇f, Hₓ², Hᵧ², H_z²", "honeydew")

    # Arrows: input -> hidden -> ... -> linear
    arrow(1.4, 1.95, 1.8, 1.95)
    arrow(3.0, 1.95, 3.4, 1.95)
    arrow(4.6, 1.95, 5.0, 1.95)
    arrow(6.2, 1.95, 6.6, 1.95)
    arrow(7.8, 1.95, 8.2, 1.95)
    arrow(9.6, 1.95, 10.0, 1.95)
    # Hessian arrows
    arrow(8.9, 1.6, 8.9, 1.1)
    arrow(9.6, 0.75, 10.0, 0.75)

    ax.text(4.2, 2.8, "10 layers (layer field) or 15 layers (toolpath field)", ha="center", fontsize=9, style="italic")
    ax.text(
        0.2, 0.4, "input  →  sinusoidal MLP  →  scalar field f(x); autograd extracts ∇f and Hessian rows", fontsize=9
    )

    _save(fig, "siren_architecture.png")


# ---------------------------------------------------------------------------
# 8. Curvature regimes (planar / spherical / saddle / cylinder)
# ---------------------------------------------------------------------------


def render_curvature_regimes() -> None:
    """Four panels showing the four canonical curvature regimes."""
    fig = plt.figure(figsize=(12, 3.5))
    titles = [
        "Planar\nK₁ = K₂ = 0",
        "Sphere (convex)\nK₁ = K₂ < 0",
        "Cylinder\nK₁ = 0, K₂ ≠ 0",
        "Saddle\nK₁ · K₂ < 0",
    ]

    def f_planar(u, v):
        return np.zeros_like(u)

    def f_sphere(u, v):
        return 0.4 - np.sqrt(np.maximum(0.4**2 - u**2 - v**2, 0))

    def f_cylinder(u, v):
        return 0.4 - np.sqrt(np.maximum(0.4**2 - u**2, 0))

    def f_saddle(u, v):
        return 0.6 * (u**2 - v**2)

    surfaces = [f_planar, f_sphere, f_cylinder, f_saddle]
    cmaps = ["bone", "viridis", "viridis", "coolwarm"]

    for idx, (fn, title, cmap) in enumerate(zip(surfaces, titles, cmaps)):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        u = np.linspace(-0.35, 0.35, 40)
        v = np.linspace(-0.35, 0.35, 40)
        U, V = np.meshgrid(u, v)
        Z = fn(U, V)
        ax.plot_surface(U, V, Z, cmap=cmap, alpha=0.85, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=-55)
        # Equal aspect-ish.
        ax.set_xlim([-0.35, 0.35])
        ax.set_ylim([-0.35, 0.35])
        ax.set_zlim([-0.2, 0.2])

    fig.suptitle("Principal-curvature regimes (with outward gradient convention, K is signed)", fontsize=11)
    fig.tight_layout()
    _save(fig, "curvature_regimes.png")


# ---------------------------------------------------------------------------
# 9. Collision weight schedule
# ---------------------------------------------------------------------------


def render_collision_schedule() -> None:
    """Show how collision-loss weights step over training epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    epochs = np.arange(0, 500)
    near = np.where(epochs > 300, 4e4, np.where(epochs > 200, 2e4, 1e4))
    far = np.where(epochs > 300, 4e4, np.where(epochs > 200, 1.6e4, 8e3))

    ax.step(epochs, near, where="post", color="crimson", linewidth=2.0, label="near cone ($_COLLISION_SCHEDULE_NEAR$)")
    ax.step(
        epochs, far, where="post", color="steelblue", linewidth=2.0, label="far / inner ($_COLLISION_SCHEDULE_FAR$)"
    )
    ax.axvline(200, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(300, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(100, 4.5e4, "early", ha="center", fontsize=9, color="dimgray")
    ax.text(250, 4.5e4, "mid", ha="center", fontsize=9, color="dimgray")
    ax.text(400, 4.5e4, "late", ha="center", fontsize=9, color="dimgray")

    ax.set_xlabel("training epoch")
    ax.set_ylabel("loss weight")
    ax.set_title("Collision-loss weight schedule\n(ramps up so the layer field does not collapse early)")
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 5e4])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    _save(fig, "collision_schedule.png")


# ---------------------------------------------------------------------------
# 10. Geodesic curvature illustration
# ---------------------------------------------------------------------------


def render_geodesic_curvature() -> None:
    """A toolpath inside a layer surface, with its tangent and geodesic
    normal vectors drawn."""
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")

    # Layer surface: a gently curved patch.
    u = np.linspace(-1, 1, 30)
    v = np.linspace(-1, 1, 30)
    U, V = np.meshgrid(u, v)
    Z = 0.15 * (U**2 + V**2)
    ax.plot_surface(U, V, Z, color="lightgray", alpha=0.4, linewidth=0.0, edgecolor="none")

    # Toolpath on the surface: a tilted spiral arc.
    t = np.linspace(-0.7, 0.7, 100)
    xt = t
    yt = 0.6 * t + 0.25 * t**2
    zt = 0.15 * (xt**2 + yt**2)
    ax.plot(xt, yt, zt, color="crimson", linewidth=2.4, label="toolpath")

    # A point partway along the toolpath, with tangent + geodesic normal.
    idx = 70
    p = np.array([xt[idx], yt[idx], zt[idx]])
    # Tangent (finite difference).
    tang = np.array([xt[idx + 1] - xt[idx - 1], yt[idx + 1] - yt[idx - 1], zt[idx + 1] - zt[idx - 1]])
    tang /= np.linalg.norm(tang)
    # Layer normal at the point (gradient of 0.15(u^2+v^2): (0.3u, 0.3v, -1)).
    n = np.array([0.3 * p[0], 0.3 * p[1], -1.0])
    n /= np.linalg.norm(n)
    # Geodesic normal = tangent × layer normal (within the layer).
    gn = np.cross(tang, n)
    gn /= np.linalg.norm(gn)

    ax.quiver(*p, *tang * 0.4, color="darkgreen", linewidth=1.6, arrow_length_ratio=0.15, label="tangent T̂")
    ax.quiver(*p, *gn * 0.4, color="steelblue", linewidth=1.6, arrow_length_ratio=0.15, label="geodesic normal")
    ax.quiver(*p, *n * 0.4, color="black", linewidth=1.0, arrow_length_ratio=0.15, label="layer normal n̂")
    ax.scatter(*p, color="black", s=50)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 0.5])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Geodesic curvature\nbending of the toolpath *within* the layer's tangent plane")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=25, azim=-60)
    _save(fig, "geodesic_curvature.png")


# ---------------------------------------------------------------------------
# 11. Architecture overview (end-to-end pipeline)
# ---------------------------------------------------------------------------


def render_architecture_overview() -> None:
    """High-level data-flow diagram of one pipeline iteration."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 5])
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0})

    # Top row: ingest
    block(0.2, 4.0, 1.8, 0.7, "mesh + stress\n(experiment_loaders)", "lightyellow")
    block(2.4, 4.0, 1.6, 0.7, "config JSON\n+ resolve_device\n+ set_global_seed", "lightyellow")

    # Models
    block(4.4, 4.0, 2.0, 0.7, "build_field_bundle\n(2× SIREN + PlatformModel)", "lavender")
    arrow(2.0, 4.35, 2.4, 4.35)
    arrow(4.0, 4.35, 4.4, 4.35)

    # Optimizers
    block(6.8, 4.0, 2.0, 0.7, "3× Adam optimizers", "lavender")
    arrow(6.4, 4.35, 6.8, 4.35)

    # Main loop
    block(
        0.4,
        2.4,
        8.6,
        1.0,
        "epoch loop\nfor batch in zipped_set: forward → losses → backward → optimizer.step()",
        "mistyrose",
    )
    arrow(7.8, 4.0, 4.7, 3.4)

    # Loss aggregation
    block(0.4, 0.9, 2.2, 1.0, "layer + collision + base + boundary", "mistyrose")
    block(3.0, 0.9, 2.2, 1.0, "+ toolpath + stress", "mistyrose")
    block(5.6, 0.9, 2.2, 1.0, "+ platform (epoch > 1900)", "mistyrose")
    block(8.2, 0.9, 1.6, 1.0, "Σ → total loss", "honeydew")
    arrow(2.6, 1.4, 3.0, 1.4)
    arrow(5.2, 1.4, 5.6, 1.4)
    arrow(7.8, 1.4, 8.2, 1.4)
    arrow(4.7, 2.4, 4.7, 1.9)

    # Outputs
    block(9.4, 4.0, 2.4, 0.7, "checkpoints + losses\n(training_outputs)", "honeydew")
    arrow(9.0, 1.4, 10.6, 1.4)
    arrow(10.6, 1.4, 10.6, 3.9)

    ax.set_title("Pipeline architecture (one of {support-free, toolpath-alignment})", fontsize=11)
    _save(fig, "architecture_overview.png")


if __name__ == "__main__":
    render_cone_sampler()
    render_tangent_circle()
    render_tool_envelope()
    render_support_angle()
    render_layer_field()
    render_loss_composition()
    render_siren_architecture()
    render_curvature_regimes()
    render_collision_schedule()
    render_geodesic_curvature()
    render_architecture_overview()
