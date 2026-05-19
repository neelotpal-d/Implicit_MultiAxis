"""Reusable loss terms for the example training pipelines."""

import torch

from shared_geometry import computeGeodesicCurvature2, computePrincipalCurvatures, supportLoss
from training_dataclasses import loss_enabled


def collision_weight(epoch: int, early: float, mid: float, late: float):
    """Piecewise collision-loss weight schedule from the original experiments."""
    if epoch > 300:
        return late
    if epoch > 200:
        return mid
    return early


def compute_layer_loss(batch_input_points, scalar_field, data, config, master_switch: str | None = None):
    """
    Loss to penalize the violation of threholds on gradient norm and curvature of layer field

    Returns the loss, a detached record value, the field output dictionary, and
    gradients so follow-up losses can reuse the same forward pass.
    """
    inps = batch_input_points
    out = scalar_field(inps)
    grads = out["grads"]
    loss = out["scalars"].sum() * 0.0
    record = loss.detach()

    if master_switch is not None and not loss_enabled(config, master_switch):
        return loss, record, out, grads

    if loss_enabled(config, "use_gradient_loss"):
        # Encourage the gradient norm to vary smoothly across the sampled volume.
        grad_norm = torch.norm(grads, dim=1)
        grad_norm_grad = torch.autograd.grad(
            grad_norm,
            inps,
            torch.ones_like(grad_norm),
            create_graph=True,
        )[0]

        grad_norm_error = torch.sum(grad_norm_grad * grad_norm_grad, dim=1) / (grad_norm + 1e-10)
        grad_norm_loss = torch.mean(grad_norm_error)

        grad_norm_square = grad_norm * grad_norm
        # Discourage near-zero gradients, which make level-set extraction unstable.
        small_grad_loss = torch.mean(torch.exp(-100 * grad_norm_square))

        gradient_loss = config.layer_gradient_weight * grad_norm_loss
        gradient_loss += config.layer_small_gradient_weight * small_grad_loss
        loss += gradient_loss
        record += gradient_loss.detach()

    if loss_enabled(config, "use_curvature_loss"):
        # Limit principal curvature after normalizing by the mesh scale.
        curvatures = computePrincipalCurvatures(
            out["HX2"],
            out["HY2"],
            out["HZ2"],
            grads,
            epsilons=config.curvature_epsilons,
        )
        curvature_losses = 30 * torch.relu(torch.abs(curvatures / data.range_vals[0]) - config.curvature_limit)
        curvature_loss = torch.mean(curvature_losses * curvature_losses)

        weighted_curvature_loss = config.layer_curvature_weight * curvature_loss
        loss += weighted_curvature_loss
        record += weighted_curvature_loss.detach()

    return loss, record, out, grads


def _accumulate_collision_term(loss, record, term, weight, name: str, epoch: int):
    """Add a collision sub-term to loss and record, warning if non-finite.

    The original code silently dropped NaN/Inf terms from ``loss`` while
    still adding them to ``record``, which polluted the log files with
    monotonically NaN values and made it impossible to spot which
    collision constraint had blown up. Here we zero both contributions
    on non-finite values *and* emit a single-line warning so the user
    knows their collision constraint is being skipped.
    """
    if not torch.isfinite(term):
        print(
            f"[warning] {name} collision loss non-finite at epoch {epoch}; this batch's contribution is skipped.",
            flush=True,
        )
        return loss, record
    return loss + weight * term, record + term


def add_collision_losses(loss, batch_input_points, out, grads, scalar_field, collision_loss, data, config, epoch: int):
    """Add near, far, and inside collision penalties for the current tool model."""
    if not loss_enabled(config, "use_collision_loss"):
        return loss, 0

    col_record = torch.zeros((), dtype=torch.float32)
    lim_vals = [data.x_lim, data.y_lim, data.z_lim]

    col_loss = collision_loss.collision_scalar_loss(
        batch_input_points, grads, out["scalars"], scalar_field, limVals=lim_vals
    )["loss"]
    loss, col_record = _accumulate_collision_term(
        loss,
        col_record,
        col_loss,
        collision_weight(epoch, 1e4, 2e4, 4e4),
        "near",
        epoch,
    )

    far_loss = collision_loss.collision_scalar_loss_far(
        batch_input_points,
        grads,
        out["scalars"],
        scalar_field,
        limVals=lim_vals,
        n_angles=10,
    )["loss"]
    loss, col_record = _accumulate_collision_term(
        loss,
        col_record,
        far_loss,
        collision_weight(epoch, 8e3, 1.6e4, 4e4),
        "far",
        epoch,
    )

    far_loss2 = collision_loss.collision_scalar_loss_far2(
        batch_input_points,
        grads,
        out["scalars"],
        scalar_field,
        limVals=lim_vals,
        n_angles=10,
    )["loss"]
    loss, col_record = _accumulate_collision_term(
        loss,
        col_record,
        far_loss2,
        collision_weight(epoch, 8e3, 1.6e4, 4e4),
        "far2",
        epoch,
    )

    if epoch > 0:
        inner_loss = collision_loss.collision_scalar_loss_far_in(
            batch_input_points,
            grads,
            out["scalars"],
            scalar_field,
            limVals=lim_vals,
        )["loss"]
        loss, col_record = _accumulate_collision_term(
            loss,
            col_record,
            inner_loss,
            collision_weight(epoch, 8e3, 1.6e4, 4e4),
            "inner",
            epoch,
        )

    return loss, col_record


def add_base_loss(loss, scalar_field, batch_base_points, config, epoch: int):
    """Encourage base points to align with the build direction."""
    if epoch <= 0 or not loss_enabled(config, "use_base_loss"):
        return loss, 0

    out = scalar_field(batch_base_points)
    target_grad = torch.tensor([0, 1, 0], device=config.device, dtype=torch.float32).unsqueeze(0)
    grad_norm = torch.norm(out["grads"], dim=1).unsqueeze(1)
    grad_error = (out["grads"] / (grad_norm + 1e-10) - target_grad) * 2.5
    grad_loss = torch.mean(grad_error * grad_error)

    return loss + 1e0 * grad_loss, grad_loss


def add_boundary_support_loss(loss, scalar_field, batch_bound_points, batch_bound_normals, config, epoch: int):
    """Penalize boundary gradients that violate the configured support angle."""
    if epoch <= 0 or not loss_enabled(config, "use_boundary_support_loss"):
        return loss, 0

    out = scalar_field(batch_bound_points)
    support_out = supportLoss(batch_bound_normals, out["grads"])
    boundary_loss = support_out["loss"]
    return loss + 1e0 * boundary_loss, boundary_loss


def compute_toolpath_loss(
    models,
    batch_input_points,
    primary_out,
    batch_min,
    batch_max,
    batch_range_lim,
    data,
    config,
    gradient_norm_weight,
    curvature_weight,
):
    """Train the toolpath field to stay tangent to the layer field with limited curvature and uniform spacing."""
    if not loss_enabled(config, "use_toolpath_loss"):
        zero = primary_out["scalars"].sum() * 0.0
        return zero, zero.detach(), zero.detach()

    field1_grads = primary_out["grads"]
    field1_scalars = primary_out["scalars"]
    field1_scalars = (2 * field1_scalars.detach() - (batch_max + batch_min)) * batch_range_lim / (batch_max - batch_min)
    field1_scalars.requires_grad = True

    out2 = models.scalar_field2(batch_input_points)
    grads2 = out2["grads"][:, 0:3]

    field1_grads_norm = torch.norm(field1_grads, dim=1).unsqueeze(1)
    field1_grads_orig = field1_grads
    field1_grads_unit = field1_grads / (field1_grads_norm + 1e-10)

    normal_comp = torch.sum(field1_grads_unit.detach() * grads2, dim=1).unsqueeze(1)
    projected_grads = grads2 - normal_comp * field1_grads_unit.detach()
    project_norm = torch.norm(projected_grads, dim=1)
    target_norm = torch.ones_like(project_norm).to(data.range_vals.device)
    grad_error = project_norm - target_norm
    grad_norm_loss = torch.mean(torch.abs(grad_error * grad_error))

    curvatures_tp = computeGeodesicCurvature2(
        field1_grads_orig,
        grads2,
        primary_out["HX2"],
        primary_out["HY2"],
        primary_out["HZ2"],
        out2["HX2"],
        out2["HY2"],
        out2["HZ2"],
    )
    curvatures_tp = 10.0 * torch.relu(torch.abs(curvatures_tp / data.range_vals[0]) - 1.0 / 5.0)

    good_norm_mask = 1 / (1 + torch.exp(-200 * (project_norm - 3e-1)))
    curvature_loss = torch.sum(curvatures_tp * good_norm_mask * curvatures_tp)
    curvature_loss = curvature_loss / torch.sum(good_norm_mask.detach())

    loss = gradient_norm_weight * grad_norm_loss + curvature_weight * curvature_loss
    return loss, loss.detach(), curvature_loss.detach()


def compute_stress_losses(
    models, batch_s_points, batch_s_dirs, batch_s_wts, batch_min, batch_max, batch_range_lim, epoch: int, config
):
    """Align layer/toolpath directions with stress directions at sampled points."""
    if epoch <= 1 or not loss_enabled(config, "use_stress_loss"):
        zero = torch.tensor(0.0, device=batch_s_points.device)
        return zero, zero, zero, zero

    primary_out = models.scalar_field(batch_s_points)
    s_grads = primary_out["grads"]
    s_grads = s_grads / (torch.norm(s_grads, dim=1).unsqueeze(1) + 1e-10)

    dotprd = torch.sum(s_grads * batch_s_dirs, dim=1)
    stress_loss = torch.mean(batch_s_wts.squeeze() * dotprd * dotprd)

    field1_grads = primary_out["grads"]
    field1_scalars = primary_out["scalars"]
    field1_scalars = (2 * field1_scalars.detach() - (batch_max + batch_min)) * batch_range_lim / (batch_max - batch_min)
    field1_scalars.requires_grad = True

    out2 = models.scalar_field2(batch_s_points)
    grads2 = out2["grads"][:, 0:3]

    field1_grads = field1_grads / (torch.norm(field1_grads, dim=1, keepdim=True) + 1e-10)
    tangents = torch.cross(field1_grads, grads2)

    cross_error = torch.cross(tangents, batch_s_dirs)
    cross_error = torch.sum(cross_error * cross_error, dim=1)
    stress_loss2 = torch.mean(batch_s_wts.squeeze() * cross_error)

    dot_prod = torch.sum(grads2 * batch_s_dirs, dim=1)
    stress_loss4 = torch.mean(batch_s_wts.squeeze() * torch.abs(dot_prod * dot_prod))

    total_loss = 1e1 * stress_loss + 1e1 * stress_loss2 + 1e1 * stress_loss4
    layer_record = 1e1 * stress_loss.detach()
    cross_record = 1e1 * stress_loss2.detach()
    tp_record = 1e1 * stress_loss4.detach()
    return total_loss, layer_record, cross_record, tp_record
