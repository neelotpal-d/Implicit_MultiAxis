import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
from sdfField import sdfModel

np.bool = np.bool_

"""
Collision and clearance losses for neural scalar fields.

The training scripts treat the gradient of a scalar field as a local build/tool
direction. This file samples a small "tool envelope" around that direction and
penalizes scalar-field orderings that would imply self-intersection, layer
collision, or tool/model interference.

The init_tool_* presets are specific to our current tool shape/size and
clearance assumptions. Change those radii and sample distances whenever the
physical tool geometry changes.
"""




def sample_directions_in_cone(directions, samples):
    """
    Generates samples within a cone of angle theta around each direction in `directions`.
    
    Args:
        directions: Tensor of shape (n, 3), each row is a unit direction vector.
        samples: Tensor of shape (m, 3) representing pre-sampled directions in the canonical frame.
    
    Returns:
        A tensor of shape (n, m, 3) containing sampled directions.
    """
    n, m = directions.shape[0], samples.shape[0]
    
    # Build a local frame for every input direction, then express the canonical
    # cone samples in that frame.
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=directions.device).expand(n, 3)
    
    # Choose an arbitrary perpendicular vector
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=directions.device).expand(n, 3)
    
    # Handle cases where direction is near (0,0,1) to avoid singularity
    close_to_x = torch.abs(directions[:, 0]) > 0.99
    # arbitrary_vector[close_to_z] = torch.tensor([0.0, 1.0, 0.0], device=directions.device)
    arbitrary_vector = arbitrary_vector.clone()  # Clone before modifying
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=directions.device)

    
    u = torch.nn.functional.normalize(torch.cross(arbitrary_vector, directions), dim=1)
    v = torch.cross(directions, u)
    
    # Rotate samples to align with the given directions
    rotated_samples = samples[:, 0:1] * u.unsqueeze(1) + \
                      samples[:, 1:2] * v.unsqueeze(1) + \
                      samples[:, 2:3] * directions.unsqueeze(1)
    
    return rotated_samples  # Shape (n, m, 3)



def get_cone_sample_direction_cosines(angle, m, device='cuda'):
    """Randomly sample directions in a canonical cone aligned with +Z."""
    theta = torch.tensor(angle * torch.pi / 180)  # Opening angle in radians

    # Precompute samples in the canonical frame
    cos_alpha = torch.rand(m) * (1 - torch.cos(theta)) + torch.cos(theta)
    sin_alpha = torch.sqrt(1 - cos_alpha**2)
    phi = torch.rand(m) * (2 * torch.pi)
    
    x = sin_alpha * torch.cos(phi)
    y = sin_alpha * torch.sin(phi)
    z = cos_alpha
    samples = torch.stack((x, y, z), dim=-1)  # Shape (m, 3)
    
    return samples.to(device)


def get_cone_sample_direction_cosines2(angle, m, device='cuda'):
    """Sample a canonical cone with extra coverage near the cone boundary."""
    theta = torch.tensor(angle * torch.pi / 180)  # Opening angle in radians
    
    # First sample: exactly along the axis
    samples = [torch.tensor([[0.0, 0.0, 1.0]], device=device)]
    
    # Second group: m/4 samples at alpha = theta/2 with uniform phi
    num_fixed_alpha = m // 4
    phi_fixed = torch.linspace(0, 2 * torch.pi, num_fixed_alpha, device=device)
    alpha_fixed = theta / 2
    cos_alpha_fixed = torch.cos(alpha_fixed)
    sin_alpha_fixed = torch.sin(alpha_fixed)

    x_fixed = sin_alpha_fixed * torch.cos(phi_fixed)
    y_fixed = sin_alpha_fixed * torch.sin(phi_fixed)
    z_fixed = cos_alpha_fixed * torch.ones_like(phi_fixed)
    
    samples.append(torch.stack((x_fixed, y_fixed, z_fixed), dim=-1))
    
    # Third group: (3m/4 - 1) samples, biased near alpha = theta
    num_remaining = m - (num_fixed_alpha + 1)
    phi_remaining = torch.rand(num_remaining, device=device) * (2 * torch.pi)
    
    # Biasing towards theta (e.g., using cos^n sampling for concentration)
    n = 5  # Adjust this exponent to control concentration
    cos_alpha_remaining = torch.cos(theta) + (1 - torch.cos(theta)) * torch.rand(num_remaining, device=device) ** n
    sin_alpha_remaining = torch.sqrt(1 - cos_alpha_remaining**2)

    x_remaining = sin_alpha_remaining * torch.cos(phi_remaining)
    y_remaining = sin_alpha_remaining * torch.sin(phi_remaining)
    z_remaining = cos_alpha_remaining
    
    samples.append(torch.stack((x_remaining, y_remaining, z_remaining), dim=-1))
    
    #print(samples)
    # Combine all samples
    samples = torch.cat(samples, dim=0)
    
    return samples.to(device)


def get_cone_sample_direction_cosines3(angle, m, device='cuda'):
    """
    Samples `m` directions within a cone of opening angle `angle` (in degrees),
    ensuring uniform azimuthal (`phi`) distribution.

    Args:
        angle (float): Cone opening angle in degrees.
        m (int): Number of samples.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Shape (m, 3), sampled direction cosines.
    """
    theta = torch.tensor(angle * torch.pi / 180, device=device)  # Opening angle in radians

    # First sample: Exactly along the axis
    samples = [torch.tensor([[0.0, 0.0, 1.0]], device=device)]

    # Second group: m/4 samples at alpha = theta/2, with uniform phi
    num_fixed_alpha = m // 4
    phi_fixed = torch.linspace(0, 2 * torch.pi, num_fixed_alpha, device=device)
    alpha_fixed = theta / 2
    cos_alpha_fixed = torch.cos(alpha_fixed)
    sin_alpha_fixed = torch.sin(alpha_fixed)

    x_fixed = sin_alpha_fixed * torch.cos(phi_fixed)
    y_fixed = sin_alpha_fixed * torch.sin(phi_fixed)
    z_fixed = cos_alpha_fixed * torch.ones_like(phi_fixed)

    samples.append(torch.stack((x_fixed, y_fixed, z_fixed), dim=-1))

    # Third group: (3m/4 - 1) samples, evenly distributed in phi
    num_remaining = m - (num_fixed_alpha + 1)
    
    # Stratify phi by spreading them evenly instead of random sampling
    phi_remaining = torch.linspace(0, 2 * torch.pi, num_remaining, device=device)

    # Biasing towards theta using cos^n sampling. The original code wrote
    # `(1 - 1) * torch.rand(...)` which collapsed every sample in this group
    # to exactly cos(theta), making the cone a 2- or 3-latitude rosette.
    # Matching the randomized branch in get_cone_sample_direction_cosines2.
    n = 5  # Adjust this exponent to control concentration
    cos_alpha_remaining = torch.cos(theta) + (1 - torch.cos(theta)) * torch.rand(num_remaining, device=device) ** n
    sin_alpha_remaining = torch.sqrt(1 - cos_alpha_remaining**2)

    x_remaining = sin_alpha_remaining * torch.cos(phi_remaining)
    y_remaining = sin_alpha_remaining * torch.sin(phi_remaining)
    z_remaining = cos_alpha_remaining

    samples.append(torch.stack((x_remaining, y_remaining, z_remaining), dim=-1))

    # Combine all samples
    samples = torch.cat(samples, dim=0)

    return samples.to(device)




def sample_points_along_directions(sampled_directions, dist_vals, origins):
    """
    Samples points along given directions at specified distances and translates them by the origins.
    
    Args:
        sampled_directions: Tensor of shape (n, m, 3) containing direction vectors.
        dist_vals: Tensor of shape (k,) containing distances to sample points along each direction.
        origins: Tensor of shape (n, 3) containing the origin points for each cone.
    
    Returns:
        A tensor of shape (n, m, k, 3) containing sampled points.
    """
    n, m, _ = sampled_directions.shape
    k = dist_vals.shape[0]
    
    # Expand dimensions to broadcast correctly
    dist_vals = dist_vals.view(1, 1, k, 1)  # Shape (1, 1, k, 1)
    sampled_directions = sampled_directions.unsqueeze(2)  # Shape (n, m, 1, 3)
    
    sampled_points = sampled_directions * dist_vals  # Shape (n, m, k, 3)
    
    # Translate points by the origins
    origins = origins.view(n, 1, 1, 3)  # Shape (n, 1, 1, 3)
    sampled_points = sampled_points + origins
    
    return sampled_points



def sample_tangent_circle(base_points, gradients, m, d):
    """
    For each point in `base_points`, sample `m` points in a uniform circle in the tangent plane at distance `d`.

    Args:
        base_points (torch.Tensor): Shape (n, 3), the original points.
        gradients (torch.Tensor): Shape (n, 3), the gradient directions (plane normal).
        m (int): Number of samples per point.
        d (float): Radius of the circle.

    Returns:
        torch.Tensor: Shape (n, m, 3), sampled points in the tangent plane.
    """
    n = base_points.shape[0]

    # The gradient is treated as the normal of the local layer surface; the
    # sampled circle approximates a cutter/nozzle footprint around the point.
    # Normalize gradients to get unit normal directions
    normal = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)  # (n, 3)

    # Create an arbitrary vector for cross product (must not be parallel to normal)
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=base_points.device).expand(n, 3).clone()
    close_to_x = (torch.abs(normal[:, 0]) > 0.9)  # Avoid collinearity with x-axis
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=base_points.device)

    # Compute first tangent vector (u) using cross product
    u = torch.cross(normal, arbitrary_vector)  # (n, 3)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Compute second tangent vector (v) using cross product
    v = torch.cross(normal, u)  # (n, 3)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Generate `m` angles uniformly in [0, 2π]
    theta = torch.linspace(0, 2 * torch.pi, m, device=base_points.device).view(1, m, 1)  # (1, m)

    # Compute circle points in local tangent basis
    circle_offsets = d * (torch.cos(theta) * u.unsqueeze(1) + torch.sin(theta) * v.unsqueeze(1))  # (n, m, 3)

    # Translate to base points
    sampled_points = base_points.unsqueeze(1) + circle_offsets  # (n, m, 3)

    return sampled_points



def sample_tangent_circle2(base_points, gradients, m, d):
    """
    For each point in `base_points`, sample `m` points in a uniform circle in the tangent plane at distance `d`.

    Args:
        base_points (torch.Tensor): Shape (n, 3), the original points.
        gradients (torch.Tensor): Shape (n, 3), the gradient directions (plane normal).
        m (int): Number of samples per point.
        d (float): Radius of the circle.

    Returns:
        torch.Tensor: Shape (n, m, 3), sampled points in the tangent plane.
    """
    n = base_points.shape[0]

    # Same footprint as sample_tangent_circle, but with a random angular offset
    # so repeated calls do not always test the same spokes.
    # Normalize gradients to get unit normal directions
    normal = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)  # (n, 3)

    # Create an arbitrary vector for cross product (must not be parallel to normal)
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=base_points.device).expand(n, 3).clone()
    close_to_x = (torch.abs(normal[:, 0]) > 0.9)  # Avoid collinearity with x-axis
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=base_points.device)

    # Compute first tangent vector (u) using cross product
    u = torch.cross(normal, arbitrary_vector)  # (n, 3)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Compute second tangent vector (v) using cross product
    v = torch.cross(normal, u)  # (n, 3)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Generate `m` angles uniformly in [0, 2π]
    # print(m)
    angle_offset = torch.pi/m
    angle_offset_randomized = (2*torch.pi/m)*np.random.rand()
    # print(type(angle_offset))
    theta = torch.linspace(angle_offset_randomized, 2 * torch.pi + angle_offset_randomized, m, device=base_points.device).view(1, m, 1)  # (1, m)

    # Compute circle points in local tangent basis
    circle_offsets = d * (torch.cos(theta) * u.unsqueeze(1) + torch.sin(theta) * v.unsqueeze(1))  # (n, m, 3)

    # Translate to base points
    sampled_points = base_points.unsqueeze(1) + circle_offsets  # (n, m, 3)

    return sampled_points


def sample_tangent_circle3(base_points, gradients, m, d):
    """
    For each point in `base_points`, sample `m` points in a uniform circle in the tangent plane at distance `d`.

    Args:
        base_points (torch.Tensor): Shape (n, 3), the original points.
        gradients (torch.Tensor): Shape (n, 3), the gradient directions (plane normal).
        m (int): Number of samples per point.
        d (float): Radius of the circle.

    Returns:
        torch.Tensor: Shape (n, m, 3), sampled points in the tangent plane.
    """
    n = base_points.shape[0]

    # Normalize gradients to get unit normal directions
    normal = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)  # (n, 3)

    # Create an arbitrary vector for cross product (must not be parallel to normal)
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], device=base_points.device).expand(n, 3).clone()
    close_to_x = (torch.abs(normal[:, 0]) > 0.9)  # Avoid collinearity with x-axis
    arbitrary_vector[close_to_x] = torch.tensor([0.0, 1.0, 0.0], device=base_points.device)

    # Compute first tangent vector (u) using cross product
    u = torch.cross(normal, arbitrary_vector)  # (n, 3)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Compute second tangent vector (v) using cross product
    v = torch.cross(normal, u)  # (n, 3)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Generate `m` angles uniformly in [0, 2π]
    # print(m)
    angle_offset = torch.pi/m
    angle_offset_randomized = (2*torch.pi/m)*np.random.rand()
    # print(type(angle_offset))
    theta = torch.linspace(angle_offset_randomized, 2 * torch.pi + angle_offset_randomized, m, device=base_points.device).view(1, m, 1)  # (1, m)

    # Compute circle points in local tangent basis
    circle_offsets = d * (torch.cos(theta) * u.unsqueeze(1) + torch.sin(theta) * v.unsqueeze(1))  # (n, m, 3)

    # Translate to base points
    sampled_points = base_points.unsqueeze(1) + circle_offsets  # (n, m, 3)

    return sampled_points


def sample_along_gradient(tangent_samples, gradients, distances):
    """
    For each point in `tangent_samples`, sample `k` points along the gradient direction.

    Args:
        tangent_samples (torch.Tensor): Shape (n, m, 3), points sampled on the tangent plane.
        gradients (torch.Tensor): Shape (n, 3), the gradient direction at each base point.
        distances (torch.Tensor): Shape (k,), distances to sample along the gradient.

    Returns:
        torch.Tensor: Shape (n, m, k, 3), sampled points along the gradient.
    """
    n, m, _ = tangent_samples.shape
    k = distances.shape[0]

    # Sweep every tangent-plane footprint point forward along the local build
    # direction to create a sparse volume/envelope test.
    # Normalize gradients to get unit direction
    grad_unit = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)  # (n, 3)

    # Compute displacement vectors for each distance
    displacement = grad_unit.unsqueeze(1).unsqueeze(2) * distances.view(1, 1, k, 1)  # (n, 1, k, 3)

    # Expand tangent_samples to (n, m, k, 3) and apply displacements
    sampled_points = tangent_samples.unsqueeze(2) + displacement  # (n, m, k, 3)

    return sampled_points





class collison_loss:
    """
    Additive-style collision loss.

    This class checks whether samples in the local forward tool envelope have
    scalar values that should already be "behind" the base point. Positive
    errors mean the learned scalar ordering would let layer/tool geometry cut
    into occupied or earlier material.
    """
    def __init__(self,sample_num, angle, device='cuda', distList = [0.05, 0.1, 0.4, 0.8], model_load_path = None):
        # Cone directions are generated once and rotated to each gradient during
        # loss evaluation; distances/radii are configured by init_tool_*.
        # IMPORTANT: the init_tool_* presets below are calibrated for our
        # current tool geometry only. If the tool shape, nozzle/cutter radius,
        # or required clearance changes, these distances and radii must be
        # changed to match the new physical envelope.
        self.sampled_directions_seed = get_cone_sample_direction_cosines3(angle, sample_num, device=device)
        self.dist_vals = torch.tensor(distList, dtype = torch.float32, device=device)
        self.sdfModel = None
        if(model_load_path):
            self.sdfModel = sdfModel(device=device, model_load_path=model_load_path)
        self.device = device
        self.dist_array_far = None
        self.radi_far = None
        self.dist_array_in = None
        self.radi_in = None
        
    def init_tool_standard(self, scale=1.0):
        """Configure a coarse, fixed tool envelope in model units."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 11.0, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        self.dist_array_far =  torch.tensor([22.5,33.75,45], dtype = torch.float32, device=self.device)
        self.radi_far = 25.0
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([7.5, 11.25, 33.75], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale
        
        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
    
        
    def init_tool_dense(self, scale=1.0):
        """Configure denser near/far sample distances for stricter clearance."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        self.dist_array_far =  torch.tensor([22.5,33.75,35,40,42.5, 45,47.5,50,55,60,65,70], dtype = torch.float32, device=self.device)
        self.radi_far = 25.0
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([4.3, 7.5, 11.25, 33.75,44.5, 49.5, 52.5, 57.5,70, 80], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale


        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
        
        
    
    def init_tool_dense_uniform1(self, scale=1.0):
        """Uniform-ish dense profile tuned in comments for fertility/clip runs."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        # self.dist_array_far =  torch.tensor([15.00, 22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5], dtype = torch.float32, device=self.device)
        # self.dist_array_far =  torch.tensor([15.00,20.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5], dtype = torch.float32, device=self.device)
        self.dist_array_far =  torch.tensor([18.00,23.00,27.5,32.5,37.5,42.5,47.5,52.5,57.5], dtype = torch.float32, device=self.device)

        self.radi_far = 24 #24.00 change it to 24 for fertility 28 for clip
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([4.5, 7.25, 11.25, 33.75, 44.5, 49.5, 52.5, 57.5], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale


        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
        
        
        
    def init_tool_dense_uniform2(self, scale=1.0):
        """Alternative uniform-ish dense profile with farther samples."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        self.dist_array_far =  torch.tensor([25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0], dtype = torch.float32, device=self.device)
        self.radi_far = 25.0
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([7.5, 11.25,33.75,44.5, 49.5, 52.5, 57.5,70, 80], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale


        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
        
        
        
    def init_tool_dense1(self, scale=1.0):
        """Sparse experimental profile with selected far and inner distances."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        self.dist_array_far =  torch.tensor([22.5,35,40,42.5,70,80,90], dtype = torch.float32, device=self.device)
        self.radi_far = 25.0
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([7.5, 11.25,33.75,44.5,60], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale


        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
        
    def init_tool_dense2(self, scale=1.0):
        """Sparse experimental profile biased toward farther checks."""
        
        self.dist_vals  = torch.tensor([3.75, 7.5, 9.75, 12.75], dtype = torch.float32, device=self.device)
        self.dist_vals = self.dist_vals/scale
        
        self.dist_array_far =  torch.tensor([33.75,45,50,55,60,65,85], dtype = torch.float32, device=self.device)
        self.radi_far = 25.0
        self.dist_array_far = self.dist_array_far/scale
        self.radi_far = self.radi_far/scale
        
        
        
        self.dist_array_in =  torch.tensor([7.5, 52.5, 57.5,70, 80], dtype = torch.float32, device=self.device)
        self.radi_in = 11.25
        self.dist_array_in = self.dist_array_in/scale
        self.radi_in = self.radi_in/scale


        
        print("--------------------------------")
        print(f"Initialising tool for a scale {scale}")
        print(f"cone samples: {self.dist_vals}")
        print(f"Far cylinder: radi({self.radi_far:.2f}); samples:({self.dist_array_far})")
        print(f"Inner cylinder: radi({self.radi_in:.2f}); samples:({self.dist_array_in})")        
        print("--------------------------------")
       
        
        
    def collision_gradient_loss(self, points, grads, scalarField, limVals = [1.0,1.0,1.0]):
        """Penalize sampled directions where the field gradient points backward."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sampled_directions = sample_directions_in_cone(grad_dirns, self.sampled_directions_seed)
        sample_points = sample_points_along_directions(sampled_directions, self.dist_vals, points)
        output_at_samples = scalarField(sample_points)
        gradient_at_samples = output_at_samples['grads']
        
        ####CAUTION: This may lead to instability#############
        
        gradient_at_samples = gradient_at_samples/ (torch.norm(gradient_at_samples, dim=1)+1e-10).unsqueeze(1)
        
        #######################################################
        #print(gradient_at_samples.shape)
        direction_at_samples = sampled_directions.unsqueeze(2)
        #print(direction_at_samples.shape)
        dot_prod = torch.sum(gradient_at_samples*direction_at_samples, dim=3)
        
        #print(dot_prod.shape)
        #select the negative ones
        error_samples = torch.relu(-dot_prod)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        error_mask = error_mask*in_mask
        
        
        
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'directions':sampled_directions, 'samples':sample_points, 'dir_at_samples':direction_at_samples, 'mask':error_mask, 'in_mask':in_mask}
    
    
    
    def collision_scalar_loss(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Cone-envelope loss based on scalar ordering along candidate tool paths."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sampled_directions = sample_directions_in_cone(grad_dirns, self.sampled_directions_seed)
        sample_points = sample_points_along_directions(sampled_directions, self.dist_vals, points)
        output_at_samples = scalarField(sample_points)
        scalar_at_samples = output_at_samples['scalars']

        #print(gradient_at_samples.shape)
        scalars_at_baseSamples = scalars.unsqueeze(1).unsqueeze(2)
        #print(direction_at_samples.shape)
        
        #print(dot_prod.shape)
        #select the negative ones
        
        
        scalars_max = torch.max(scalars)
        scalars_min = torch.min(scalars)
        # Tiny compensation avoids zero-margin degeneracy when sampled values
        # are numerically equal to the base scalar.
        scalar_comp = 1e-4*(scalars_max - scalars_min)
        
        errors = scalars_at_baseSamples - scalar_at_samples + scalar_comp.detach()
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        
        
        

        if(self.sdfModel):
            # Optional learned SDF mask keeps the loss focused on samples inside
            # the modeled part/valid domain.
            in_mask2 = self.limFun(sample_points)
            assert in_mask.shape == in_mask2.shape
            in_mask = in_mask*in_mask2
            
        # print("--------------")
        # print(in_mask)
        # print("-------------")
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'directions':sampled_directions, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
        
    def collision_mixed_loss(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Combine gradient-facing and scalar-ordering checks in the cone envelope."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sampled_directions = sample_directions_in_cone(grad_dirns, self.sampled_directions_seed)
        sample_points = sample_points_along_directions(sampled_directions, self.dist_vals, points)
        output_at_samples = scalarField(sample_points)
        gradient_at_samples = output_at_samples['grads']
        scalars_at_samples = output_at_samples['scalars']
        ####CAUTION: This may lead to instability#############
        
        gradient_at_samples = gradient_at_samples/ (torch.norm(gradient_at_samples, dim=1)+1e-10).unsqueeze(1)
        
        #######################################################
        #print(gradient_at_samples.shape)
        direction_at_samples = sampled_directions.unsqueeze(2)
        #print(direction_at_samples.shape)
        dot_prod = torch.sum(gradient_at_samples*direction_at_samples, dim=3)
        #print(dot_prod.shape)
        #select the negative ones
        error_samples = torch.relu(-dot_prod)
        
        ####
        scalars_at_base = scalars.unsqueeze(1).unsqueeze(2)
        #print(scalars_at_base.shape)
        #print(scalars_at_samples.shape)
        
        scalar_diffs = scalars_at_base - scalars_at_samples
        ##Adding compensate term to offset the boundary a bit to remove degenerecies:
        scalars_max = torch.max(scalars)
        scalars_min = torch.min(scalars)
        scalar_comp = 0.02*(scalars_max - scalars_min)
        ##
        scalar_diffs = scalar_diffs + scalar_comp.detach()
        in_range_mask = scalar_diffs[...,0] > 0
        ####
        
        #print(in_range_mask.shape)
        
        
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(in_mask.shape)
        error_mask = error_mask*in_mask*in_range_mask

        ms_error = torch.mean(error_samples[in_mask*in_range_mask]*error_samples[in_mask*in_range_mask])
        if(torch.sum(in_mask*in_range_mask)==0):
            ms_error = 0
        #To use a mask later
        
        
        return {'loss':ms_error, 'directions':sampled_directions, 'samples':sample_points, 'dir_at_samples':direction_at_samples, 'mask':error_mask, 'in_mask':in_mask}
    
    
    
    def collision_scalar_loss_far(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0], dist_array = [0.3,0.45,0.6], radi_ = 0.3, n_angles=10):
        """Check a cylindrical/ring-like envelope farther from the base point."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        radi = 0
        if(self.radi_far is not None):
            radi = self.radi_far
        else:
            radi = radi_
            
        sample_points_at_base =  sample_tangent_circle(points, grads, n_angles, radi) #n,m,3 used 0.15
        
        sample_distances = None
        if(self.dist_array_far is not None):
            sample_distances = self.dist_array_far
        else:
            sample_distances = torch.tensor(dist_array, device=self.device) #k, used 0.3, 0.4, 0.6
            
            
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        scalar_at_samples = scalarField(sample_points)['scalars']
        scalars_at_baseSamples = scalars.unsqueeze(1).unsqueeze(2)
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        
        scalars_max = torch.max(scalars)
        scalars_min = torch.min(scalars)
        scalar_comp = 2e-4*(scalars_max - scalars_min)
        
        errors = scalars_at_baseSamples - scalar_at_samples + scalar_comp.detach()
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

        if(self.sdfModel):
            in_mask2 = self.limFun(sample_points)
            assert in_mask.shape == in_mask2.shape
            in_mask = in_mask*in_mask2
    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    
    
    
    def collision_scalar_loss_far2(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0], dist_array = [0.3,0.45,0.6], radi_ = 0.3, n_angles=10):
        """Far envelope variant with randomized circle phase and distances."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        radi = 0
        if(self.radi_far is not None):
            radi = self.radi_far
        else:
            radi = radi_
            
        sample_points_at_base =  sample_tangent_circle2(points, grads, n_angles, radi) #n,m,3 used 0.15
        
        sample_distances = None
        if(self.dist_array_far is not None):
            # print("---")
            # print(self.dist_array_far)
            sample_distances = self.dist_array_far + 0.75*(self.dist_array_far[1] - self.dist_array_far[0])*torch.rand(self.dist_array_far.shape, device=self.device)
            # print(sample_distances)
        else:
            sample_distances = torch.tensor(dist_array, device=self.device) #k, used 0.3, 0.4, 0.6
            
            
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        scalar_at_samples = scalarField(sample_points)['scalars']
        scalars_at_baseSamples = scalars.unsqueeze(1).unsqueeze(2)
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        
        scalars_max = torch.max(scalars)
        scalars_min = torch.min(scalars)
        scalar_comp = 2e-4*(scalars_max - scalars_min)
        
        errors = scalars_at_baseSamples - scalar_at_samples + scalar_comp.detach()
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

        if(self.sdfModel):
            in_mask2 = self.limFun(sample_points)
            assert in_mask.shape == in_mask2.shape
            in_mask = in_mask*in_mask2
    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
        
    
    def collision_scalar_loss_far_in(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0], dist_array = [0.1,0.15,0.45], radi_ = 0.15):
        """Inner-radius envelope check for closer material/tool interference."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        radi = 0
        if(self.radi_in is not None):
            radi = self.radi_in
        else:
            radi = radi_
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, radi) #n,m,3 used 0.15
        
        sample_distances = None
        if(self.dist_array_in is not None):
            sample_distances = self.dist_array_in
        else:
            sample_distances = torch.tensor(dist_array, device=self.device) #k, used 0.3, 0.4, 0.6
        
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        scalar_at_samples = scalarField(sample_points)['scalars']
        scalars_at_baseSamples = scalars.unsqueeze(1).unsqueeze(2)
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        errors = scalars_at_baseSamples - scalar_at_samples
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

        if(self.sdfModel):
            in_mask2 = self.limFun(sample_points)
            assert in_mask.shape == in_mask2.shape
            in_mask = in_mask*in_mask2
    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    def limModel(self, points, limVals):
        """Axis-aligned normalized-domain mask used before averaging errors."""
        x_points = torch.abs(points[...,0])
        y_points = torch.abs(points[...,1])
        z_points = torch.abs(points[...,2])
        
        
        mask_x = x_points<limVals[0]
        mask_y = y_points<limVals[1]
        mask_z = z_points<limVals[2]
        
        
        mask = mask_x*mask_y*mask_z
        
        return mask
    
    
    def limFun(self, points):
        """Optional learned-SDF mask; true where samples are inside the SDF model."""
        inMask = True*torch.ones_like(points)
        if(self.sdfModel):
            outVals = self.sdfModel.predictOuts(points)
            outScalars = outVals['scalars']
            inMask = outScalars<0
            assert inMask.shape[-1] == 1
        inMask = inMask[...,0]
        
        return inMask
    
    
    






class collison_loss_milling:
    """
    Milling-style collision loss.

    These checks use the same local footprint sampling, but the SDF terms look
    for tool samples that enter the solid model or violate already-defined
    scalar layers from a subtractive/manufacturing-clearance perspective.
    """
    def __init__(self,sample_num, angle, device='cuda', distList = [0.05, 0.1, 0.4, 0.8], model_load_path = None):
        self.sampled_directions_seed = get_cone_sample_direction_cosines3(angle, sample_num)
        self.dist_vals = torch.tensor(distList, dtype = torch.float32, device=device)
        self.sdfModel = None
        if(model_load_path):
            self.sdfModel = sdfModel(model_load_path = model_load_path)
        self.device = device  
        
    
    def collision_scalar_loss_far_in(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Milling variant that compares sampled scalars just beyond the base layer."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, 0.025) #n,m,3 used 0.15
        sample_distances = torch.tensor([0.1,0.2,0.4,0.5],device=self.device) #k, used 0.3, 0.4, 0.6
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        scalar_at_samples = scalarField(sample_points)['scalars']
        scalars_at_baseSamples = scalars.unsqueeze(1).unsqueeze(2)
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        errors = scalar_at_samples - scalars_at_baseSamples
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        
        #adjust inmask to exclude points inside model as surfaces are not defined there
        
    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    
    
    def collision_scalar_loss_in_layer1(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Penalize sampled points that violate layer ordering outside the model."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, 0.085) #n,m,3 used 0.15
        sample_distances = torch.tensor([0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.5,0.55,0.6,0.7,0.8,1.0],device=self.device) #k, used 0.3, 0.4, 0.6
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        sdf_at_samples = self.sdfModel.predictOuts(sample_points)['scalars']
        
        
        scalar_at_samples = scalarField(sample_points)['scalars']
        scalars_at_baseSamples = scalars.view(-1,1,1,1)
        
        # print("++++++++++++++++++")
        # print(scalar_at_samples.shape)
        # print(scalars_at_baseSamples.shape)
        # print("++++++++++++++++++")
              
        errors = scalars_at_baseSamples - scalar_at_samples
        # print(errors.shape)
        error_samples = torch.relu(10*errors)
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask
        if(self.sdfModel):
            in_mask2 = self.limFun2(sample_points)
            assert in_mask.shape == in_mask2.shape
            in_mask = in_mask*in_mask2
    
        error_mask = error_mask[...,0]*in_mask
        # print("------")
        # print(error_samples.shape)
        # print(in_mask.shape)
        # print(error_samples[in_mask].shape)
        # print("------")
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    
    
    def collision_scalar_loss_in_model1(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Small-footprint SDF penetration penalty."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, 0.085) #n,m,3 used 0.15
        sample_distances = torch.tensor([0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.5,0.55,0.6,0.7,0.8,1.0],device=self.device) #k, used 0.3, 0.4, 0.6
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        sdf_at_samples = self.sdfModel.predictOuts(sample_points)['scalars']
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        errors = torch.relu(-sdf_at_samples)
        error_samples = 10*errors
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    
    def collision_scalar_loss_in_model2(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Medium-footprint SDF penetration penalty."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, 0.29) #n,m,3 used 0.15
        sample_distances = torch.tensor([0.9,1.1,1.2,1.5,1.6,1.7,1.8],device=self.device) #k, used 0.3, 0.4, 0.6
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        sdf_at_samples = self.sdfModel.predictOuts(sample_points)['scalars']
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        errors = torch.relu(-sdf_at_samples)
        error_samples = 10*errors
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    
    
    
    def collision_scalar_loss_in_model3(self, points, grads, scalars, scalarField, limVals = [1.0,1.0,1.0]):
        """Large-footprint SDF penetration penalty."""
        grad_norms = torch.norm(grads, dim=1).unsqueeze(1)
        grad_dirns = grads/(grad_norms + 1e-10)
        
        sample_points_at_base =  sample_tangent_circle(points, grads, 10, 1.00) #n,m,3 used 0.15
        sample_distances = torch.tensor([1.9,2.0,2.1,2.2,2.3],device=self.device) #k, used 0.3, 0.4, 0.6
        sample_points = sample_along_gradient(sample_points_at_base, grads, sample_distances) #n,m,k,3

        sdf_at_samples = self.sdfModel.predictOuts(sample_points)['scalars']
        
        #print(scalar_at_samples.shape)
        #print(scalars_at_baseSamples.shape)
        errors = torch.relu(-sdf_at_samples)
        error_samples = 10*errors
        error_mask = error_samples > 0
        in_mask = self.limModel(sample_points, limVals)
        #print(error_samples.shape)
        #print(in_mask.shape)
        
        #error_mask = error_mask[...,0]*in_mask

    
        error_mask = error_mask[...,0]*in_mask
        ms_error = torch.mean(error_samples[in_mask]*error_samples[in_mask])
        
        #To use a mask later
        
        
        return {'loss':ms_error, 'samples':sample_points,  'mask':error_mask, 'in_mask':in_mask}
    
    def limModel(self, points, limVals):
        """Axis-aligned normalized-domain mask used before averaging errors."""
        x_points = torch.abs(points[...,0])
        y_points = torch.abs(points[...,1])
        z_points = torch.abs(points[...,2])
        
        
        mask_x = x_points<limVals[0]
        mask_y = y_points<limVals[1]
        mask_z = z_points<limVals[2]
        
        
        mask = mask_x*mask_y*mask_z
        
        return mask
    
    
    def limFun(self, points):
        """SDF mask for samples inside the solid model."""
        inMask = True*torch.ones_like(points)
        if(self.sdfModel):
            outVals = self.sdfModel.predictOuts(points)
            outScalars = outVals['scalars']
            inMask = outScalars<0
            assert inMask.shape[-1] == 1
        inMask = inMask[...,0]
        
        return inMask
    
    def limFun2(self, points):
        """SDF mask for samples outside the solid model."""
        inMask = True*torch.ones_like(points)
        if(self.sdfModel):
            outVals = self.sdfModel.predictOuts(points)
            outScalars = outVals['scalars']
            inMask = outScalars>0
            assert inMask.shape[-1] == 1
        inMask = inMask[...,0]
        
        return inMask
        
