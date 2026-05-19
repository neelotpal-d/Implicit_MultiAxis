"""Render algorithm diagrams matched to the paper's aesthetic.

The Dutta et al. 2026 paper uses a recognisable visual style: muted
teal / dusty rose / warm gold palette, real VTK-rendered 3D meshes,
clean sans-serif labels, no decorative noise. This script reproduces
that aesthetic by:

- using **PyVista** (the project's existing dep) for 3D illustrations —
  matplotlib's 3D backend cannot match VTK-rendered surfaces;
- using **matplotlib** for the 2D data plots, with a Helvetica /
  paper-muted-palette restyling;
- composing PyVista screenshots into matplotlib subplots when the
  figure has multiple panels (curvature regimes).

Output goes into ``docs/figures/`` as PNG. Run via
``pixi run render-diagrams``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pv.OFF_SCREEN = True


# ---------------------------------------------------------------------------
# Paper-matched palette (sampled from the Dutta et al. teaser figure)
# ---------------------------------------------------------------------------

INK = "#2a2a2a"          # body text + axes
TEAL = "#5a8d8a"         # primary accent (the paper's signature teal)
ROSE = "#c0908e"         # secondary accent (the dusty-pink)
GOLD = "#c4a574"         # tertiary (warm tan)
BLUE_MUTED = "#7a9bb8"   # subtle blue
PANEL = "#a8a8a8"        # panel border gray
MESH = "#dadada"         # neutral mesh material
GRID = "#ececec"
MUTED = "#7a7a7a"

# Field colormap that matches the paper's red->orange->yellow->teal->dark-red
# gradient. We synthesise this from named matplotlib stops so we don't depend
# on a custom colormap file.
PAPER_FIELD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "paper_field",
    [
        (0.0, "#9c1b1b"),   # deep red — high
        (0.25, "#d97a5a"),  # warm orange
        (0.5, "#e7c468"),   # yellow-gold
        (0.7, "#7ab0a8"),   # teal
        (1.0, "#3d6d6a"),   # deep teal — low
    ],
)


mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.facecolor": "white",
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
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.8,
    "lines.markersize": 8.0,
    "lines.markeredgewidth": 0.0,
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


def _pv_plotter(size: tuple[int, int] = (900, 900)) -> pv.Plotter:
    """A paper-styled PyVista plotter — white background, soft lighting."""
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("white")
    return p


def _add_panel_frame(ax) -> None:
    """Draw the paper's rounded-rectangle panel border around an axes."""
    from matplotlib.patches import FancyBboxPatch

    # In axes-fraction coordinates, slightly inside the bounds.
    rect = FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor=PANEL,
        facecolor="none",
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# 1. Cone direction sampler  (PyVista)
# ---------------------------------------------------------------------------


def render_cone_sampler() -> None:
    p = _pv_plotter(size=(900, 900))

    angle_deg = 60.0
    theta = np.radians(angle_deg)
    height = 1.0

    # Soft semi-transparent cone — paper-style "geometry-as-context".
    # PyVista's Cone has its apex at `center + (height/2) * direction`. We
    # want apex at the origin (0,0,0) opening upward toward the rim at z=1,
    # so direction = (0, 0, -1) with center = (0, 0, height/2).
    cone = pv.Cone(center=(0, 0, height / 2), direction=(0, 0, -1),
                   height=height, radius=height * np.tan(theta), resolution=80)
    p.add_mesh(cone, color=TEAL, opacity=0.22, smooth_shading=True,
               show_edges=False)
    # Outline circle at the rim.
    rim_pts = []
    for phi in np.linspace(0, 2 * np.pi, 240):
        rim_pts.append([height * np.tan(theta) * np.cos(phi),
                        height * np.tan(theta) * np.sin(phi), height])
    rim = pv.lines_from_points(np.array(rim_pts), close=True)
    p.add_mesh(rim, color=TEAL, line_width=2.5)
    # Three meridians.
    for phi_m in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        meridian = pv.Line(
            (0, 0, 0),
            (np.tan(theta) * np.cos(phi_m), np.tan(theta) * np.sin(phi_m), height),
        )
        p.add_mesh(meridian, color=TEAL, line_width=1.6, opacity=0.7)

    # Axis sample.
    axis_pt = pv.Sphere(radius=0.04, center=(0, 0, 1.0))
    p.add_mesh(axis_pt, color=INK, smooth_shading=True)

    # Half-angle ring.
    m = 24
    n_ring = m // 4
    cos_a = np.cos(theta / 2)
    sin_a = np.sin(theta / 2)
    for k in range(n_ring):
        phi = 2 * np.pi * k / n_ring
        s = pv.Sphere(radius=0.035, center=(sin_a * np.cos(phi),
                                            sin_a * np.sin(phi), cos_a))
        p.add_mesh(s, color=GOLD, smooth_shading=True)

    # Boundary-biased tail.
    rng = np.random.default_rng(0)
    n_tail = m - n_ring - 1
    for k in range(n_tail):
        phi = 2 * np.pi * k / n_tail
        cos_alpha = np.cos(theta) + (1 - np.cos(theta)) * rng.uniform() ** 5
        sin_alpha = np.sqrt(1 - cos_alpha**2)
        s = pv.Sphere(radius=0.032, center=(sin_alpha * np.cos(phi),
                                            sin_alpha * np.sin(phi), cos_alpha))
        p.add_mesh(s, color=ROSE, smooth_shading=True)

    # Cone-axis arrow — tip ABOVE the rim so it reads as "build direction".
    arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1.35,
                     tip_length=0.20, tip_radius=0.04, shaft_radius=0.012,
                     tip_resolution=40, shaft_resolution=40)
    p.add_mesh(arrow, color=INK, smooth_shading=True)

    # Camera with 20% margin around the geometry — fits on the page.
    p.camera_position = [(3.8, -3.8, 2.4), (0, 0, 0.6), (0, 0, 1)]
    p.camera.zoom(0.82)
    snap = OUT_DIR / "_cone_raw.png"
    p.screenshot(str(snap), transparent_background=False)
    p.close()

    # Compose with matplotlib so we get Helvetica labels + legend + frame.
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    img = plt.imread(snap)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Legend (made of legend handles, not real lines).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK,  markersize=10, label="axis  (1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GOLD, markersize=10, label=fr"$\alpha = \theta/2$ ring  ({n_ring})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROSE, markersize=10, label=fr"boundary-biased  ({n_tail})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
    ax.text(0.95, 0.05, r"$\nabla f$", fontsize=15, color=INK,
            transform=ax.transAxes, ha="right")
    _add_panel_frame(ax)
    _save(fig, "cone_sampler.png")
    snap.unlink()


# ---------------------------------------------------------------------------
# 2. Tangent-plane circle sampler  (PyVista)
# ---------------------------------------------------------------------------


def render_tangent_circle() -> None:
    p = _pv_plotter(size=(900, 750))

    origin = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.22, 0.30, 1.0]); normal /= np.linalg.norm(normal)

    arbitrary = np.array([1.0, 0.0, 0.0])
    u = np.cross(arbitrary, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Tangent disk.
    disk = pv.Disc(center=origin, normal=normal, inner=0.0, outer=0.70,
                   r_res=2, c_res=120)
    p.add_mesh(disk, color=TEAL, opacity=0.18, smooth_shading=True,
               show_edges=False)
    # Disk outline.
    out_pts = []
    for phi in np.linspace(0, 2 * np.pi, 240):
        out_pts.append(origin + 0.70 * np.cos(phi) * u + 0.70 * np.sin(phi) * v)
    p.add_mesh(pv.lines_from_points(np.array(out_pts), close=True),
               color=TEAL, line_width=2.5)

    # Circle samples.
    m = 12
    radius = 0.48
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    ring_pts = []
    for phi in angles:
        pt = origin + radius * np.cos(phi) * u + radius * np.sin(phi) * v
        ring_pts.append(pt)
        s = pv.Sphere(radius=0.04, center=pt)
        p.add_mesh(s, color=ROSE, smooth_shading=True)
    p.add_mesh(pv.lines_from_points(np.array(ring_pts), close=True),
               color=ROSE, line_width=1.5, opacity=0.6)

    # Normal arrow.
    arrow = pv.Arrow(start=origin, direction=normal, scale=0.95,
                     tip_length=0.14, tip_radius=0.045, shaft_radius=0.014)
    p.add_mesh(arrow, color=INK, smooth_shading=True)

    # Origin marker.
    p.add_mesh(pv.Sphere(radius=0.045, center=origin), color=INK,
               smooth_shading=True)

    p.camera_position = [(2.0, -2.2, 1.5), (0, 0, 0.3), (0, 0, 1)]
    p.camera.zoom(1.0)
    snap = OUT_DIR / "_circle_raw.png"
    p.screenshot(str(snap), transparent_background=False)
    p.close()

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.imshow(plt.imread(snap))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROSE, markersize=10, label=fr"$m={m}$ tangent samples"),
        Line2D([0], [0], color=TEAL, lw=2.5, label="tangent plane"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
    ax.text(0.62, 0.78, r"$\hat n_1$", fontsize=15, color=INK,
            transform=ax.transAxes)
    _add_panel_frame(ax)
    _save(fig, "tangent_circle.png")
    snap.unlink()


# ---------------------------------------------------------------------------
# 3. Tool envelope (2D matplotlib)
# ---------------------------------------------------------------------------


def render_tool_envelope() -> None:
    from collisionLoss import TOOL_PROFILES

    profile = TOOL_PROFILES["dense_uniform1"]
    fig, ax = plt.subplots(figsize=(5.6, 7.2))

    cone_angle = np.radians(30.0)
    z_cone = np.linspace(0, max(profile.dist_vals), 50)
    r_cone = z_cone * np.tan(cone_angle)
    ax.fill_betweenx(z_cone, -r_cone, r_cone, color=ROSE, alpha=0.22, edgecolor="none")
    ax.scatter(np.zeros(len(profile.dist_vals)), profile.dist_vals,
               color=ROSE, s=46, zorder=5, edgecolor=INK, linewidth=0.5)

    ax.fill_betweenx(profile.dist_array_in,
                     -profile.radi_in, profile.radi_in,
                     color=GOLD, alpha=0.22, edgecolor="none")
    for d in profile.dist_array_in:
        ax.scatter([-profile.radi_in, profile.radi_in], [d, d],
                   color=GOLD, s=32, zorder=5, edgecolor=INK, linewidth=0.5)

    ax.fill_betweenx(profile.dist_array_far,
                     -profile.radi_far, profile.radi_far,
                     color=TEAL, alpha=0.22, edgecolor="none")
    for d in profile.dist_array_far:
        ax.scatter([-profile.radi_far, profile.radi_far], [d, d],
                   color=TEAL, s=32, zorder=4, edgecolor=INK, linewidth=0.5)

    ax.annotate("", xy=(0, 7), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.0,
                            "mutation_scale": 18})
    ax.text(1.6, 4.0, r"$\nabla f$", fontsize=15, color=INK)
    ax.scatter([0], [0], color=INK, s=80, zorder=11)
    ax.text(-2.5, -2.5, "base point", fontsize=10, color=MUTED)

    label_x = profile.radi_far + 2
    ax.text(label_x, profile.dist_array_far[len(profile.dist_array_far) // 2],
            "far cylinder\n" + fr"$r = {profile.radi_far:.0f}$ mm",
            color=TEAL, fontsize=10, va="center")
    ax.text(label_x, profile.dist_array_in[len(profile.dist_array_in) // 2],
            "inner cylinder\n" + fr"$r = {profile.radi_in:.2f}$ mm",
            color=GOLD, fontsize=10, va="center")
    ax.text(label_x, profile.dist_vals[len(profile.dist_vals) // 2],
            "near cone\n" + r"half-angle $30^\circ$",
            color=ROSE, fontsize=10, va="center")

    ax.set_xlim([-profile.radi_far - 4, profile.radi_far + 18])
    ax.set_ylim([-6, max(profile.dist_array_far) + 8])
    ax.set_xlabel("radial offset  (mesh mm)")
    ax.set_ylabel(r"distance along $\nabla f$  (mesh mm)")
    ax.set_aspect("equal")
    _save(fig, "tool_envelope.png")


# ---------------------------------------------------------------------------
# 4. Support-angle hinge loss (2D)
# ---------------------------------------------------------------------------


def render_support_angle() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    angles_deg = np.linspace(0, 180, 400)
    threshold_deg = 132.0
    sharpness = 25.0

    cos_thr = np.cos(np.radians(threshold_deg))
    hinge = np.maximum(0.0, sharpness * (-np.cos(np.radians(angles_deg)) + cos_thr))
    loss = hinge**2

    ax.fill_between(angles_deg, 0, loss, color=ROSE, alpha=0.22)
    ax.plot(angles_deg, loss, color=ROSE, linewidth=2.0)
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
# 5. Layer field — 2D level sets (paper colormap)
# ---------------------------------------------------------------------------


def render_layer_field() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    grid = np.linspace(-1.5, 1.5, 240)
    xv, yv = np.meshgrid(grid, grid)
    field = yv + 0.45 * np.exp(-3 * xv**2) * np.cos(1.6 * yv)

    im = ax.imshow(field, origin="lower", extent=(-1.5, 1.5, -1.5, 1.5),
                   cmap=PAPER_FIELD_CMAP, aspect="equal", alpha=0.95)
    cs = ax.contour(xv, yv, field, levels=15, colors="white",
                    linewidths=0.8, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f", colors="white")

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
# 6. Curvature regimes — 4 PyVista renders composed via matplotlib
# ---------------------------------------------------------------------------


def render_curvature_regimes() -> None:
    """Four canonical regimes, rendered in PyVista and composed in matplotlib."""

    # Camera shared by all four panels so the series reads consistently.
    # 20% margin baked in via the zoom.
    cam = [(1.05, -1.4, 0.85), (0, 0, 0.05), (0, 0, 1)]

    def render_one(mesh, *, cmap_name: str | None, color: str | None,
                   tag: str, zoom: float = 1.5) -> Path:
        p = _pv_plotter(size=(500, 500))
        if cmap_name is not None:
            p.add_mesh(mesh, cmap=cmap_name, smooth_shading=True,
                       show_edges=False, show_scalar_bar=False,
                       ambient=0.22, diffuse=0.82, specular=0.18,
                       specular_power=12)
        else:
            p.add_mesh(mesh, color=color, smooth_shading=True,
                       show_edges=False,
                       ambient=0.22, diffuse=0.82, specular=0.18,
                       specular_power=12)
        p.camera_position = cam
        p.camera.zoom(zoom * 0.85)
        snap = OUT_DIR / f"_{tag}_raw.png"
        p.screenshot(str(snap), transparent_background=False)
        p.close()
        return snap

    # --- planar: just a flat disk with a single muted color.
    planar = pv.Disc(center=(0, 0, 0), inner=0.0, outer=0.32,
                     normal=(0, 0, 1), r_res=2, c_res=120)
    snap_planar = render_one(planar, cmap_name=None, color=MESH, tag="planar")

    # --- sphere: real sphere, upper cap, coloured by z.
    sphere = pv.Sphere(radius=0.32, theta_resolution=80, phi_resolution=80,
                       start_phi=0, end_phi=110)
    sphere = sphere.translate((0, 0, -0.1), inplace=False)
    sphere.point_data["height"] = sphere.points[:, 2]
    snap_sphere = render_one(sphere, cmap_name="viridis", color=None, tag="sphere")

    # --- cylinder: axis along x, upper half kept so the camera sees the
    #               curved surface from above with curvature in y/z plane.
    cyl = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0),
                      radius=0.30, height=0.7, resolution=160, capping=False)
    # Clip away the lower hemisphere; we want z >= 0.
    cyl_top = cyl.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=False)
    cyl_top.point_data["height"] = cyl_top.points[:, 2]
    snap_cylinder = render_one(cyl_top, cmap_name="plasma", color=None,
                                tag="cylinder")

    # --- saddle: parametric quadratic — the grid form is fine here.
    u = np.linspace(-0.4, 0.4, 80); v = np.linspace(-0.4, 0.4, 80)
    U, V = np.meshgrid(u, v)
    Z = 0.7 * (U**2 - V**2)
    saddle = pv.StructuredGrid(U, V, Z)
    saddle.point_data["height"] = Z.ravel()
    snap_saddle = render_one(saddle, cmap_name="coolwarm", color=None,
                             tag="saddle")

    snaps = [
        ("planar",   snap_planar),
        ("sphere",   snap_sphere),
        ("cylinder", snap_cylinder),
        ("saddle",   snap_saddle),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    titles = {
        "planar":   (r"planar",   r"$K_1 = K_2 = 0$"),
        "sphere":   (r"sphere",   r"$K_1 = K_2 < 0$"),
        "cylinder": (r"cylinder", r"$K_1 = 0,\ K_2 \neq 0$"),
        "saddle":   (r"saddle",   r"$K_1 \cdot K_2 < 0$"),
    }
    labels = ("a", "b", "c", "d")
    for ax, (name, snap), label in zip(axes, snaps, labels):
        ax.imshow(plt.imread(snap))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        title, sub = titles[name]
        ax.set_title(f"{title}\n{sub}", fontsize=11)
        ax.text(0.04, 0.92, f"({label})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK)
        _add_panel_frame(ax)
        snap.unlink()
    fig.tight_layout()
    _save(fig, "curvature_regimes.png")


# ---------------------------------------------------------------------------
# 7. Collision-loss weight schedule (2D)
# ---------------------------------------------------------------------------


def render_collision_schedule() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    epochs = np.arange(0, 500)
    near = np.where(epochs > 300, 4e4, np.where(epochs > 200, 2e4, 1e4))
    far = np.where(epochs > 300, 4e4, np.where(epochs > 200, 1.6e4, 8e3))

    ax.step(epochs, near, where="post", color=ROSE, linewidth=2.4, label="near cone")
    ax.step(epochs, far,  where="post", color=TEAL, linewidth=2.4, label="far / inner")
    ax.axvline(200, color=MUTED, linestyle="--", linewidth=0.7)
    ax.axvline(300, color=MUTED, linestyle="--", linewidth=0.7)
    for x, text in [(100, "early"), (250, "mid"), (400, "late")]:
        ax.text(x, 4.7e4, text, ha="center", color=MUTED)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("loss weight")
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 5.3e4])
    ax.legend(loc="lower right")
    _save(fig, "collision_schedule.png")


# ---------------------------------------------------------------------------
# 8. Geodesic curvature (PyVista surface + path)
# ---------------------------------------------------------------------------


def render_geodesic_curvature() -> None:
    p = _pv_plotter(size=(900, 720))

    # Layer patch (parametric).
    u = np.linspace(-1, 1, 60); v = np.linspace(-1, 1, 60)
    U, V = np.meshgrid(u, v)
    Z = 0.15 * (U**2 + V**2)
    surface = pv.StructuredGrid(U, V, Z)
    surface.point_data["scalar"] = Z.ravel()
    p.add_mesh(surface, cmap=PAPER_FIELD_CMAP, smooth_shading=True,
               opacity=0.95, show_edges=False, show_scalar_bar=False,
               ambient=0.2, diffuse=0.85, specular=0.15)

    # Toolpath on the surface.
    t = np.linspace(-0.7, 0.7, 200)
    xt, yt = t, 0.6 * t + 0.25 * t**2
    zt = 0.15 * (xt**2 + yt**2) + 0.01
    path_pts = np.column_stack([xt, yt, zt])
    p.add_mesh(pv.lines_from_points(path_pts), color=INK, line_width=4.0)

    # Reference frame at the marked point.
    idx = 130
    pt = np.array([xt[idx], yt[idx], zt[idx]])
    tang = np.array([xt[idx + 1] - xt[idx - 1],
                     yt[idx + 1] - yt[idx - 1],
                     zt[idx + 1] - zt[idx - 1]])
    tang /= np.linalg.norm(tang)
    n_layer = np.array([0.3 * pt[0], 0.3 * pt[1], -1.0]); n_layer /= np.linalg.norm(n_layer)
    n_geo = np.cross(tang, n_layer); n_geo /= np.linalg.norm(n_geo)

    # Bigger arrows so the local frame reads at thumbnail size.
    for direction, colour in [(tang, INK), (n_geo, GOLD), (-n_layer, ROSE)]:
        p.add_mesh(pv.Arrow(start=pt, direction=direction, scale=0.55,
                            tip_length=0.22, tip_radius=0.07,
                            shaft_radius=0.022,
                            tip_resolution=40, shaft_resolution=40),
                   color=colour, smooth_shading=True)
    p.add_mesh(pv.Sphere(radius=0.045, center=pt), color=INK,
               smooth_shading=True)

    p.camera_position = [(2.9, -3.4, 1.8), (0, 0, 0.15), (0, 0, 1)]
    p.camera.zoom(0.9)
    snap = OUT_DIR / "_geo_raw.png"
    p.screenshot(str(snap), transparent_background=False)
    p.close()

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(plt.imread(snap))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=INK,  lw=3, label=r"toolpath / $\hat T$"),
        Line2D([0], [0], color=GOLD, lw=3, label=r"geodesic normal  $\hat n_g$"),
        Line2D([0], [0], color=ROSE, lw=3, label=r"layer normal  $\hat n_1$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
    _add_panel_frame(ax)
    _save(fig, "geodesic_curvature.png")
    snap.unlink()


if __name__ == "__main__":
    render_cone_sampler()
    render_tangent_circle()
    render_tool_envelope()
    render_support_angle()
    render_layer_field()
    render_curvature_regimes()
    render_collision_schedule()
    render_geodesic_curvature()
