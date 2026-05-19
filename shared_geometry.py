import numpy as np
import torch


def supportLoss(surfaceNormals, surfaceGrads, angle_degrees=132.0, sharpness=25.0):
    """Penalize boundary gradients that violate the support-angle threshold."""
    gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
    surfaceGrads = surfaceGrads / (gradNorm + 1e-10)
    dotProd = surfaceNormals * surfaceGrads
    dotProd = torch.sum(dotProd, dim=1)

    supportError = -dotProd + np.cos(np.deg2rad(angle_degrees))
    supportMask = torch.relu(supportError) > 0.0
    supportError = torch.relu(sharpness * supportError)
    supportMask = supportError > 0.0
    supportLoss = torch.mean(supportError * supportError)
    return {"loss": supportLoss, "mask": supportMask}


def computeGaussianCurvature(dx2, dy2, dz2, grads):
    """Compute Gaussian curvature from Hessian rows and field gradients."""
    fxx = dx2[:, 0]
    fxy = dx2[:, 1]
    fxz = dx2[:, 2]
    fyy = dy2[:, 1]
    fyz = dy2[:, 2]
    fzz = dz2[:, 2]

    h11 = fyy * fzz - fyz * fyz
    h12 = fyz * fxz - fxy * fzz
    h13 = fxy * fyz - fyy * fxz
    h22 = fxx * fzz - fxz * fxz
    h23 = fxy * fxz - fxx * fyz
    h33 = fxx * fyy - fxy * fxy

    fx = grads[:, 0]
    fy = grads[:, 1]
    fz = grads[:, 2]
    norm_gradF = torch.norm(grads, dim=1)

    Kg_num = fx * fx * h11 + fy * fy * h22 + fz * fz * h33
    Kg_num = Kg_num + 2 * h12 * fx * fy + 2 * h13 * fx * fz + 2 * h23 * fy * fz
    Kg_den = norm_gradF * norm_gradF * norm_gradF * norm_gradF + 1e-10
    return (Kg_num / Kg_den).unsqueeze(1)


def computeMeanCurvature(dx2, dy2, dz2, grads):
    """Compute mean curvature from Hessian rows and field gradients."""
    fxx = dx2[:, 0]
    fxy = dx2[:, 1]
    fxz = dx2[:, 2]
    fyy = dy2[:, 1]
    fyz = dy2[:, 2]
    fzz = dz2[:, 2]

    fx = grads[:, 0]
    fy = grads[:, 1]
    fz = grads[:, 2]
    norm_gradF = torch.norm(grads, dim=1)

    Km_num = fx * fx * fxx + fy * fy * fyy + fz * fz * fzz
    Km_num = Km_num + 2 * fxy * fx * fy + 2 * fxz * fx * fz + 2 * fyz * fy * fz
    trace_h = fxx + fyy + fzz
    Km_num = Km_num - norm_gradF * norm_gradF * trace_h
    Km_den = 2 * norm_gradF * norm_gradF * norm_gradF + 1e-10
    return (Km_num / Km_den).unsqueeze(1)


def computePrincipalCurvatures(dx2, dy2, dz2, grads, epsilons=(1e-7, 2e-6, 1e-5)):
    """Compute principal curvatures with configurable discriminant fallbacks."""
    Kg = computeGaussianCurvature(dx2, dy2, dz2, grads)
    Km = computeMeanCurvature(dx2, dy2, dz2, grads)

    discriminant = Km * Km - Kg
    K1 = Km + torch.sqrt(discriminant + epsilons[0])
    K2 = Km - torch.sqrt(discriminant + epsilons[0])

    if ((discriminant + epsilons[0]) < 0).any():
        K1 = Km + torch.sqrt(discriminant + epsilons[1])
        K2 = Km - torch.sqrt(discriminant + epsilons[1])

        if ((discriminant + epsilons[1]) < 0).any():
            K1 = Km + torch.sqrt(discriminant + epsilons[2])
            K2 = Km - torch.sqrt(discriminant + epsilons[2])

    return torch.hstack((K1, K2))


def getPointInsideMask(points, x_lim=1.0, y_lim=1.0, z_lim=1.0):
    """Return a mask for points inside normalized axis-aligned limits."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    check1 = abs(x) < x_lim
    check2 = abs(y) < y_lim
    check3 = abs(z) < z_lim
    return check1 & check2 & check3


def computeGeodesicCurvature(grads1, grads2, inps):
    """Compute geodesic curvature by differentiating the tangent direction."""
    tangent = torch.cross(grads1, grads2)
    tangent_unit = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-10)

    dTx = torch.autograd.grad(
        tangent_unit[:, 0],
        inps,
        torch.ones_like(tangent_unit[:, 0]),
        create_graph=True,
    )[0]
    dTy = torch.autograd.grad(
        tangent_unit[:, 1],
        inps,
        torch.ones_like(tangent_unit[:, 1]),
        create_graph=True,
    )[0]
    dTz = torch.autograd.grad(
        tangent_unit[:, 2],
        inps,
        torch.ones_like(tangent_unit[:, 2]),
        create_graph=True,
    )[0]

    accn_x = torch.sum(dTx * tangent_unit, dim=1, keepdim=True)
    accn_y = torch.sum(dTy * tangent_unit, dim=1, keepdim=True)
    accn_z = torch.sum(dTz * tangent_unit, dim=1, keepdim=True)
    accn = torch.hstack((accn_x, accn_y, accn_z))

    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + 1e-10)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)


def computeGeodesicCurvature2(grads1, grads2, f1H2X, f1H2Y, f1H2Z, f2H2X, f2H2Y, f2H2Z):
    """Compute geodesic curvature from two gradients and their Hessian rows."""
    vector = torch.cross(grads1, grads2)
    vector_norm = torch.norm(vector, dim=1, keepdim=True) + 2e-10

    tangent = vector / vector_norm

    dVx = torch.cross(f1H2X, grads2) + torch.cross(grads1, f2H2X)
    dVy = torch.cross(f1H2Y, grads2) + torch.cross(grads1, f2H2Y)
    dVz = torch.cross(f1H2Z, grads2) + torch.cross(grads1, f2H2Z)

    dT_x = dVx / vector_norm - vector * (torch.sum(vector * dVx, dim=1, keepdim=True)) / (vector_norm**3)
    dT_y = dVy / vector_norm - vector * (torch.sum(vector * dVy, dim=1, keepdim=True)) / (vector_norm**3)
    dT_z = dVz / vector_norm - vector * (torch.sum(vector * dVz, dim=1, keepdim=True)) / (vector_norm**3)

    Kx = dT_x[:, 0] * tangent[:, 0] + dT_y[:, 0] * tangent[:, 1] + dT_z[:, 0] * tangent[:, 2]
    Ky = dT_x[:, 1] * tangent[:, 0] + dT_y[:, 1] * tangent[:, 1] + dT_z[:, 1] * tangent[:, 2]
    Kz = dT_x[:, 2] * tangent[:, 0] + dT_y[:, 2] * tangent[:, 1] + dT_z[:, 2] * tangent[:, 2]

    accn = torch.hstack((Kx.unsqueeze(1), Ky.unsqueeze(1), Kz.unsqueeze(1)))

    normal = grads1 / (torch.norm(grads1, dim=1, keepdim=True) + 1e-10)
    projected_accn = accn - torch.sum(accn * normal, dim=1, keepdim=True) * normal
    return torch.norm(projected_accn, dim=1)
