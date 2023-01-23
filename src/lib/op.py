import torch
import torch.nn.functional as F


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def calc_interp_latents(xT, cond=None, spherical=True, n_steps=8):
    alpha = torch.linspace(0, 1, n_steps).to(xT[0].device).unsqueeze(1)
    cond_interp = None
    if cond is not None:
        cond_interp = cond[0][None] * (1 - alpha) + cond[1][None] * alpha

    if spherical:
        theta = torch.arccos(cos(xT[0], xT[1]))
        x_shape = xT[0].shape
        xT_interp = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].view(-1)[None] + torch.sin(alpha[:, None] * theta) * xT[1].view(-1)[None]) / torch.sin(theta)
        xT_interp = xT_interp.view(-1, *x_shape)
    else:
        alpha = alpha.unsqueeze(2).unsqueeze(3)
        xT_interp = xT[0][None] * alpha + xT[1][None] * (1 - alpha)
    return xT_interp, cond_interp


def calc_grid_latents(zs, conds=None, spherical=True, n_steps=8):
    grid_xT, grid_cond = [], []
    xT_interp1, cond_interp1 = calc_interp_latents(
        zs[:2], None if conds is None else conds[:2], spherical, n_steps)
    xT_interp2, cond_interp2 = calc_interp_latents(
        zs[2:4], None if conds is None else conds[2:4], spherical, n_steps)
    for i in range(n_steps):
        xT, cond = calc_interp_latents(
            [xT_interp1[i], xT_interp2[i]],
            None if conds is None else [cond_interp1[i], cond_interp2[i]],
            n_steps)
        grid_xT.append(xT)
        grid_cond.append(cond)
    grid_xT = torch.stack(grid_xT)
    if conds is not None:
        grid_cond = torch.stack(grid_cond)
    return grid_xT, grid_cond


"""Utilities for learned textual inversion."""
import torch, os
import torch.nn.functional as F


def bu(img, size, align_corners=True):
    """Bilinear interpolation with Pytorch.

    Args:
      img : a list of tensors or a tensor.
    """
    if isinstance(img, list):
        return [
            F.interpolate(i, size=size, mode="bilinear", align_corners=align_corners)
            for i in img
        ]
    return F.interpolate(img, size=size, mode="bilinear", align_corners=align_corners)

