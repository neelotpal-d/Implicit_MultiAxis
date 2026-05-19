"""Render the algorithm diagrams in Nature-publication style.

Typography:    Helvetica body, STIX (math) for formulas via mathtext.
Color:         Tableau-derived saturated palette; viridis for sequential.
Line weight:   1.8pt main / 0.8pt axes, marker size 8pt.

Eight figures into ``docs/figures/`` — block diagrams live inline as
Mermaid in the markdown.

Run via ``pixi run render-diagrams``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Nature-style matplotlib defaults
# ---------------------------------------------------------------------------

INK = "#1a1a1a"
RED = "#d62728"          # Tableau red
BLUE = "#1f77b4"         # Tableau blue
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
TEAL = "#17becf"
GOLD = "#bcbd22"
GRID = "#e8e8e8"
MUTED = "#6b6b6b"

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.facecolor": "white",
    # Helvetica body — the literal Nature/Science/IEEE typeface — with
    # STIX math via mathtext for inline LaTeX-style symbols.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.titlesize": 12,
    "axes.titleweight": "regular",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Visual style.
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 8.0,
    "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.8,
    "patch.edgecolor": INK,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "axes.titlepad": 12,
    "axes.labelpad": 6,
})


def _save(fig, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def _style_3d(ax) -> None:
    """Strip the default 3D pane background; keep the gridlines faint."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "-"


# ---------------------------------------------------------------------------
# 1. Cone direction sampler
# ---------------------------------------------------------------------------


def render_cone_sampler() -> None:
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(projection="3d")
    _style_3d(ax)

    angle_deg = 60.0
    theta = np.radians(angle_deg)

    # Shaded cone surface — provides a real depth cue.
    h = np.linspace(0.02, 1.0, 32)
    phi = np.linspace(0, 2 * np.pi, 80)
    H, P = np.meshgrid(h, phi)
    Xs = H * np.tan(theta) * np.cos(P)
    Ys = H * np.tan(theta) * np.sin(P)
    Zs = H
    ax.plot_surface(Xs, Ys, Zs, color=BLUE, alpha=0.10, edgecolor="none",
                    rstride=1, cstride=1, antialiased=True)
    # Outline meridians and rim.
    rim_phi = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.tan(theta) * np.cos(rim_phi), np.tan(theta) * np.sin(rim_phi),
            np.ones_like(rim_phi), color=BLUE, linewidth=1.2)
    for phi_m in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        ax.plot([0, np.tan(theta) * np.cos(phi_m)],
                [0, np.tan(theta) * np.sin(phi_m)],
                [0, 1], color=BLUE, linewidth=0.8, alpha=0.7)

    # Group 1: axis sample.
    ax.scatter([0], [0], [1.0], color=INK, s=90, label="axis (1)", zorder=10,
               depthshade=False, edgecolors=INK)

    # Group 2: half-angle ring (m/4 samples).
    m = 24
    n_ring = m // 4
    phi_ring = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    cos_a = np.cos(theta / 2)
    sin_a = np.sin(theta / 2)
    ax.scatter(sin_a * np.cos(phi_ring), sin_a * np.sin(phi_ring),
               np.full_like(phi_ring, cos_a), color=ORANGE, s=64, zorder=8,
               depthshade=False, label=fr"$\alpha = \theta/2$ ring  ($m/4={n_ring}$)")

    # Group 3: boundary-biased tail (cos^n).
    rng = np.random.default_rng(0)
    n_tail = m - n_ring - 1
    phi_tail = np.linspace(0, 2 * np.pi, n_tail, endpoint=False)
    cos_tail = np.cos(theta) + (1 - np.cos(theta)) * rng.uniform(size=n_tail) ** 5
    sin_tail = np.sqrt(1 - cos_tail**2)
    ax.scatter(sin_tail * np.cos(phi_tail), sin_tail * np.sin(phi_tail), cos_tail,
               color=RED, s=54, zorder=7, depthshade=False,
               label=fr"boundary-biased ($\cos^5$, $n_{{tail}}={n_tail}$)")

    # Cone axis arrow.
    ax.quiver(0, 0, 0, 0, 0, 1.1, color=INK, linewidth=2.0,
              arrow_length_ratio=0.08)
    ax.text(0.08, 0.08, 1.22, r"$\nabla f$", fontsize=15)

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1.32])
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.0))
    ax.view_init(elev=20, azim=-58)
    _save(fig, "cone_sampler.png")


# ---------------------------------------------------------------------------
# 2. Tangent-plane circle sampler
# ---------------------------------------------------------------------------


def render_tangent_circle() -> None:
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(projection="3d")
    _style_3d(ax)

    origin = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.22, 0.30, 1.0])
    normal /= np.linalg.norm(normal)

    arbitrary = np.array([1.0, 0.0, 0.0])
    u = np.cross(arbitrary, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Tangent plane indicated by a soft filled ellipse + outline.
    plane_t = np.linspace(0, 2 * np.pi, 200)
    plane_r = 0.70
    plane_outline = (plane_r * np.cos(plane_t)[:, None] * u
                     + plane_r * np.sin(plane_t)[:, None] * v)
    # Build a tri-surface of the plane for shading.
    plane_grid_r = np.linspace(0, plane_r, 8)
    pg_R, pg_T = np.meshgrid(plane_grid_r, plane_t)
    pg_pts = (pg_R * np.cos(pg_T))[..., None] * u + (pg_R * np.sin(pg_T))[..., None] * v
    ax.plot_surface(pg_pts[..., 0], pg_pts[..., 1], pg_pts[..., 2],
                    color=BLUE, alpha=0.10, edgecolor="none", antialiased=True)
    ax.plot(plane_outline[:, 0], plane_outline[:, 1], plane_outline[:, 2],
            color=BLUE, linewidth=1.4)

    # Circle samples.
    m = 12
    radius = 0.48
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pts = (radius * np.cos(angles)[:, None] * u
           + radius * np.sin(angles)[:, None] * v).T
    ax.plot(np.append(pts[0], pts[0, 0]), np.append(pts[1], pts[1, 0]),
            np.append(pts[2], pts[2, 0]), color=RED, linewidth=1.2, alpha=0.7)
    ax.scatter(pts[0], pts[1], pts[2], color=RED, s=70, zorder=10,
               depthshade=False, label=fr"$m={m}$ circle samples")

    # Layer normal arrow.
    ax.quiver(*origin, *normal * 0.9, color=INK, linewidth=2.2,
              arrow_length_ratio=0.10)
    ax.text(*(origin + normal * 1.05), r"$\hat n_1$", fontsize=15)
    # Origin marker.
    ax.scatter(*origin, color=INK, s=80, zorder=12, depthshade=False)
    ax.text(0.08, 0.08, -0.08, "base point", fontsize=10, color=MUTED)

    ax.set_xlim([-0.75, 0.75]); ax.set_ylim([-0.75, 0.75]); ax.set_zlim([-0.3, 1.05])
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.0))
    ax.view_init(elev=14, azim=-62)
    _save(fig, "tangent_circle.png")


# ---------------------------------------------------------------------------
# 3. Tool envelope cross-section
# ---------------------------------------------------------------------------


def render_tool_envelope() -> None:
    """Cross-section through the gradient axis of the dense_uniform1 profile."""
    from collisionLoss import TOOL_PROFILES

    profile = TOOL_PROFILES["dense_uniform1"]
    fig, ax = plt.subplots(figsize=(5.6, 7.2))

    cone_angle = np.radians(30.0)
    z_cone = np.linspace(0, max(profile.dist_vals), 50)
    r_cone = z_cone * np.tan(cone_angle)
    ax.fill_betweenx(z_cone, -r_cone, r_cone, color=RED, alpha=0.15,
                     edgecolor="none")
    ax.scatter(np.zeros(len(profile.dist_vals)), profile.dist_vals,
               color=RED, s=46, zorder=5, edgecolor=INK, linewidth=0.6)

    ax.fill_betweenx(profile.dist_array_in,
                     -profile.radi_in, profile.radi_in,
                     color=ORANGE, alpha=0.16, edgecolor="none")
    for d in profile.dist_array_in:
        ax.scatter([-profile.radi_in, profile.radi_in], [d, d],
                   color=ORANGE, s=32, zorder=5,
                   edgecolor=INK, linewidth=0.6)

    ax.fill_betweenx(profile.dist_array_far,
                     -profile.radi_far, profile.radi_far,
                     color=BLUE, alpha=0.13, edgecolor="none")
    for d in profile.dist_array_far:
        ax.scatter([-profile.radi_far, profile.radi_far], [d, d],
                   color=BLUE, s=32, zorder=4,
                   edgecolor=INK, linewidth=0.6)

    # Build-direction arrow.
    ax.annotate("", xy=(0, 7), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.0,
                            "mutation_scale": 18})
    ax.text(1.6, 4.0, r"$\nabla f$", fontsize=15)
    # Base point.
    ax.scatter([0], [0], color=INK, s=85, zorder=11)
    ax.text(-2.5, -2.5, "base point", fontsize=10, color=MUTED, ha="left")

    # Labels (positioned right of each band).
    label_x = profile.radi_far + 2
    ax.text(label_x, profile.dist_array_far[len(profile.dist_array_far) // 2],
            "far cylinder\n" + fr"$r = {profile.radi_far:.0f}$ mm",
            color=BLUE, fontsize=10, va="center")
    ax.text(label_x, profile.dist_array_in[len(profile.dist_array_in) // 2],
            "inner cylinder\n" + fr"$r = {profile.radi_in:.2f}$ mm",
            color=ORANGE, fontsize=10, va="center")
    ax.text(label_x, profile.dist_vals[len(profile.dist_vals) // 2],
            "near cone\n" + r"half-angle $30^\circ$",
            color=RED, fontsize=10, va="center")

    ax.set_xlim([-profile.radi_far - 4, profile.radi_far + 18])
    ax.set_ylim([-6, max(profile.dist_array_far) + 8])
    ax.set_xlabel("radial offset  (mesh mm)")
    ax.set_ylabel(r"distance along $\nabla f$  (mesh mm)")
    ax.set_aspect("equal")
    _save(fig, "tool_envelope.png")


# ---------------------------------------------------------------------------
# 4. Support-angle hinge loss
# ---------------------------------------------------------------------------


def render_support_angle() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    angles_deg = np.linspace(0, 180, 400)
    threshold_deg = 132.0
    sharpness = 25.0

    cos_thr = np.cos(np.radians(threshold_deg))
    hinge = np.maximum(0.0, sharpness * (-np.cos(np.radians(angles_deg)) + cos_thr))
    loss = hinge**2

    ax.fill_between(angles_deg, 0, loss, color=RED, alpha=0.15)
    ax.plot(angles_deg, loss, color=RED, linewidth=2.0)
    ax.axvline(threshold_deg, color=INK, linestyle="--", linewidth=1.0)
    ax.text(threshold_deg + 2, max(loss) * 0.55,
            fr"$\theta_s = {threshold_deg:.0f}^\circ$",
            fontsize=12, color=INK)

    ax.set_xlabel(r"angle between $\hat n$ and $\nabla f$ (deg)")
    ax.set_ylabel("per-point loss")
    ax.set_xlim([0, 180])
    ax.set_ylim(bottom=0)
    _save(fig, "support_angle.png")


# ---------------------------------------------------------------------------
# 5. Layer field — 2D level sets
# ---------------------------------------------------------------------------


def render_layer_field() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    grid = np.linspace(-1.5, 1.5, 240)
    xv, yv = np.meshgrid(grid, grid)
    field = yv + 0.45 * np.exp(-3 * xv**2) * np.cos(1.6 * yv)

    # Shaded background.
    im = ax.imshow(field, origin="lower", extent=(-1.5, 1.5, -1.5, 1.5),
                   cmap="viridis", aspect="equal", alpha=0.85)
    # Level sets on top.
    cs = ax.contour(xv, yv, field, levels=15, colors="white",
                    linewidths=1.0, alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # Gradient field arrows.
    sg = np.linspace(-1.3, 1.3, 9)
    xs, ys = np.meshgrid(sg, sg)
    fx = -2.7 * xs * np.exp(-3 * xs**2) * np.cos(1.6 * ys)
    fy = 1 - 1.6 * 0.45 * np.exp(-3 * xs**2) * np.sin(1.6 * ys)
    mag = np.sqrt(fx**2 + fy**2)
    ax.quiver(xs, ys, fx / mag, fy / mag, color="white", scale=20,
              width=0.005, edgecolor=INK, linewidth=0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"layer scalar  $f(x, y)$", labelpad=4)
    cbar.outline.set_visible(False)

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$y$  (build axis)")
    _save(fig, "layer_field.png")


# ---------------------------------------------------------------------------
# 6. Curvature regimes
# ---------------------------------------------------------------------------


def render_curvature_regimes() -> None:
    """Four canonical regimes with shaded surfaces."""
    fig = plt.figure(figsize=(12, 3.6))
    titles = [
        (r"planar"   + "\n" + r"$K_1 = K_2 = 0$",       lambda u, v: np.zeros_like(u),                                          "Greys"),
        (r"sphere"   + "\n" + r"$K_1 = K_2 < 0$",       lambda u, v: 0.4 - np.sqrt(np.maximum(0.4**2 - u**2 - v**2, 0)),         "viridis"),
        (r"cylinder" + "\n" + r"$K_1 = 0,\ K_2 \neq 0$", lambda u, v: 0.4 - np.sqrt(np.maximum(0.4**2 - u**2, 0)),                "plasma"),
        (r"saddle"   + "\n" + r"$K_1 \cdot K_2 < 0$",   lambda u, v: 0.6 * (u**2 - v**2),                                        "coolwarm"),
    ]
    labels = ("a", "b", "c", "d")

    u = np.linspace(-0.35, 0.35, 80)
    v = np.linspace(-0.35, 0.35, 80)
    U, V = np.meshgrid(u, v)
    light = LightSource(azdeg=315, altdeg=45)

    for idx, (title, fn, cmap) in enumerate(titles):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        _style_3d(ax)
        Z = fn(U, V)
        # Shaded colouring with a directional light — gives the surfaces
        # real "form", not flat colormap blobs.
        rgb = light.shade(Z, cmap=plt.get_cmap(cmap), vert_exag=2.0,
                          blend_mode="soft")
        ax.plot_surface(U, V, Z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_title(title, fontsize=11, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlim([-0.35, 0.35]); ax.set_ylim([-0.35, 0.35]); ax.set_zlim([-0.2, 0.2])
        ax.view_init(elev=24, azim=-58)
        ax.text2D(0.02, 0.92, f"({labels[idx]})", transform=ax.transAxes,
                  fontsize=12, fontweight="bold", color=INK)
        ax.set_box_aspect((1, 1, 1))

    fig.tight_layout()
    _save(fig, "curvature_regimes.png")


# ---------------------------------------------------------------------------
# 7. Collision-loss weight schedule
# ---------------------------------------------------------------------------


def render_collision_schedule() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    epochs = np.arange(0, 500)
    near = np.where(epochs > 300, 4e4, np.where(epochs > 200, 2e4, 1e4))
    far = np.where(epochs > 300, 4e4, np.where(epochs > 200, 1.6e4, 8e3))

    ax.step(epochs, near, where="post", color=RED, linewidth=2.2,
            label="near cone")
    ax.step(epochs, far, where="post", color=BLUE, linewidth=2.2,
            label="far / inner")
    ax.axvline(200, color=MUTED, linestyle="--", linewidth=0.8)
    ax.axvline(300, color=MUTED, linestyle="--", linewidth=0.8)
    ax.text(100, 4.7e4, "early", ha="center", color=MUTED)
    ax.text(250, 4.7e4, "mid",   ha="center", color=MUTED)
    ax.text(400, 4.7e4, "late",  ha="center", color=MUTED)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("loss weight")
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 5.3e4])
    ax.legend(loc="lower right")
    _save(fig, "collision_schedule.png")


# ---------------------------------------------------------------------------
# 8. Geodesic curvature illustration
# ---------------------------------------------------------------------------


def render_geodesic_curvature() -> None:
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")
    _style_3d(ax)

    # Layer patch — shaded for depth.
    u = np.linspace(-1, 1, 80); v = np.linspace(-1, 1, 80)
    U, V = np.meshgrid(u, v)
    Z = 0.15 * (U**2 + V**2)
    light = LightSource(azdeg=315, altdeg=45)
    rgb = light.shade(Z, cmap=plt.get_cmap("viridis"), vert_exag=1.5,
                      blend_mode="soft")
    ax.plot_surface(U, V, Z, rstride=2, cstride=2, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False, alpha=0.85)

    # Toolpath on the surface.
    t = np.linspace(-0.7, 0.7, 200)
    xt, yt = t, 0.6 * t + 0.25 * t**2
    zt = 0.15 * (xt**2 + yt**2)
    ax.plot(xt, yt, zt + 0.005, color=RED, linewidth=2.5)

    # Tangent / geodesic-normal / layer-normal at a point.
    idx = 130
    p = np.array([xt[idx], yt[idx], zt[idx] + 0.005])
    tang = np.array([xt[idx + 1] - xt[idx - 1],
                     yt[idx + 1] - yt[idx - 1],
                     zt[idx + 1] - zt[idx - 1]])
    tang /= np.linalg.norm(tang)
    n = np.array([0.3 * p[0], 0.3 * p[1], -1.0]); n /= np.linalg.norm(n)
    gn = np.cross(tang, n); gn /= np.linalg.norm(gn)

    ax.quiver(*p, *tang * 0.42, color=INK,    linewidth=2.0, arrow_length_ratio=0.18)
    ax.quiver(*p, *gn   * 0.42, color=ORANGE, linewidth=2.0, arrow_length_ratio=0.18)
    ax.quiver(*p, *n    * 0.42, color="white", linewidth=2.0, arrow_length_ratio=0.18)
    ax.scatter(*p, color=INK, s=55, depthshade=False, zorder=10)

    ax.text(*(p + tang * 0.52), r"$\hat T$",   fontsize=14, color=INK)
    ax.text(*(p + gn   * 0.52), r"$\hat n_g$", fontsize=14, color=ORANGE)
    ax.text(*(p + n    * 0.52), r"$\hat n_1$", fontsize=14, color="white")

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 0.55])
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_box_aspect((2, 2, 1.1))
    ax.view_init(elev=22, azim=-58)
    _save(fig, "geodesic_curvature.png")


if __name__ == "__main__":
    render_cone_sampler()
    render_tangent_circle()
    render_tool_envelope()
    render_support_angle()
    render_layer_field()
    render_curvature_regimes()
    render_collision_schedule()
    render_geodesic_curvature()
