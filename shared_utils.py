import igl
import numpy as np
import pyvista as pv
import torch


def load_mesh_pair(surface_mesh_path, volume_mesh_path):
    """Load the surface and volume meshes used by a training run."""
    return pv.read(surface_mesh_path), pv.read(volume_mesh_path)


def mesh_vertices_faces_bounds(mesh):
    """Return mesh vertices, triangular faces, and axis-aligned bounds."""
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:4]
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return vertices, faces, mins, maxs


def build_sdf_masked_grid(mesh_vertices, mesh_faces, mins, maxs, padding, spacing, keep_distance):
    """Build a structured grid and keep points within an SDF distance threshold."""
    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs

    x_vals = np.arange(x_min - padding, x_max + padding, spacing)
    y_vals = np.arange(y_min - padding, y_max + padding, spacing)
    z_vals = np.arange(z_min - padding, z_max + padding, spacing)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    grid = pv.StructuredGrid(X.flatten(), Y.flatten(), Z.flatten())
    sdf = igl.signed_distance(grid.points, mesh_vertices, mesh_faces)[0]
    return pv.PolyData(grid.points[sdf < keep_distance])


def normalization_tensors(mins, maxs, device):
    """Create midpoint and range tensors for normalized model coordinates."""
    min_vals = torch.tensor(mins, dtype=torch.float32, device=device)
    max_vals = torch.tensor(maxs, dtype=torch.float32, device=device)
    mid_vals = 0.5 * (min_vals + max_vals)
    max_range = np.max(maxs - mins)
    range_vals = 0.5 * torch.tensor([max_range, max_range, max_range], dtype=torch.float32, device=device)
    return mid_vals, range_vals


def normalize_points(points, mid_vals, range_vals, requires_grad=True):
    """Normalize points and return a detached tensor ready for autograd."""
    points = (points - mid_vals) / range_vals
    points = points.detach()
    points.requires_grad = requires_grad
    return points


def shuffled_tensor_from_points(points, max_points, device, requires_grad=True):
    """Shuffle a point array, truncate it, and convert it to a float tensor."""
    order = torch.randperm(points.shape[0]).cpu().numpy()
    selected_points = points[order][:max_points]
    return torch.tensor(
        selected_points,
        dtype=torch.float32,
        device=device,
        requires_grad=requires_grad,
    )
