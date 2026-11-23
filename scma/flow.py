from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn


def distance_to_metal(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, (0, 1)).all():
        raise ValueError("expected a two-dimensional binary metal mask")
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(mask == 0).astype(np.float32)


def dynamic_weights(
    mask: Tensor,
    distance: Tensor,
    t: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    sigma_max: float = 128.0,
    sigma_min: float = 16.0,
) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape != distance.shape:
        raise ValueError("mask and pixel distances must have the same [B, 1, H, W] shape")
    if t.shape != (mask.shape[0],):
        raise ValueError("t must have shape [B]")
    if not all(map(math.isfinite, (alpha, beta, sigma_max, sigma_min))):
        raise ValueError("weighting parameters must be finite")
    if alpha < 0 or beta < 0 or not (sigma_max > sigma_min > 0):
        raise ValueError("expected nonnegative amplitudes and sigma_max > sigma_min > 0")
    t = t.float().view(-1, 1, 1, 1)
    width = (1 - t) * sigma_max + t * sigma_min
    neighborhood = torch.exp(-0.5 * (distance.float() / width).square())
    return torch.where(mask.bool(), 1.0 + alpha, 1.0 + beta * neighborhood)


def training_loss(
    model: nn.Module,
    clean: Tensor,
    condition: Tensor,
    mask: Tensor,
    distance: Tensor,
    *,
    t: Optional[Tensor] = None,
    noise: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    sigma_max: float = 128.0,
    sigma_min: float = 16.0,
    eps: float = 1e-8,
) -> Tensor:
    if condition.shape != clean.shape or mask.shape != clean.shape:
        raise ValueError("clean images, LI conditions, and metal masks must have matching shapes")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")
    if t is None:
        t = torch.rand(clean.shape[0], device=clean.device)
    if noise is None:
        noise = torch.randn_like(clean)
    if noise.shape != clean.shape:
        raise ValueError("noise must have the same shape as the clean images")
    weights = dynamic_weights(mask, distance, t, alpha=alpha, beta=beta,
                              sigma_max=sigma_max, sigma_min=sigma_min)
    time = t.view(-1, 1, 1, 1)
    state = (1.0 - time) * noise + time * clean
    predicted_velocity = model(state, condition, t)
    error = predicted_velocity.float() - (clean.float() - noise.float())
    numerator = (weights * error.square()).sum(dim=(1, 2, 3))
    denominator = weights.sum(dim=(1, 2, 3)) + eps
    return (numerator / denominator).mean()


@torch.no_grad()
def sample(
    model: nn.Module,
    condition: Tensor,
    *,
    correct_endpoint: Optional[Callable[[Tensor], Tensor]] = None,
    steps: int = 50,
    omega: float = 0.5,
    noise: Optional[Tensor] = None,
) -> Tensor:
    if steps < 1 or not 0.0 <= omega < math.inf:
        raise ValueError("expected a positive step count and a finite nonnegative omega")
    state = torch.randn_like(condition) if noise is None else noise
    if state.shape != condition.shape or condition.ndim != 4 or condition.shape[1] != 1:
        raise ValueError("condition and initial noise must have matching [B, 1, H, W] shapes")
    step_size = 1.0 / steps
    for k in range(steps):
        tau = k / steps
        time = torch.full((state.shape[0],), tau, dtype=torch.float32, device=state.device)
        velocity = model(state, condition, time)
        next_state = state + step_size * velocity
        if correct_endpoint is not None:
            endpoint = state + (1.0 - tau) * velocity
            corrected = correct_endpoint(endpoint)
            if corrected.shape != endpoint.shape:
                raise ValueError("PCC must return one corrected endpoint per input image")
            next_state = next_state + omega * (corrected - endpoint)
        state = next_state
    return state
