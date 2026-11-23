import math
from numbers import Integral
from typing import Protocol

import torch
from torch import Tensor


class TensorProjectionPair(Protocol):
    def forward(self, image: Tensor) -> Tensor: ...

    def adjoint(self, sinogram: Tensor) -> Tensor: ...


@torch.no_grad()
def solve_pcc_torch(endpoint: Tensor, measured: Tensor, trace: Tensor,
                    projector: TensorProjectionPair, *, rho: float = 1.0,
                    max_iterations: int = 10, relative_tolerance: float = 1e-4) -> Tensor:
    if endpoint.ndim != 2 or not endpoint.is_floating_point():
        raise ValueError("endpoint must be a 2D floating-point attenuation image")
    if measured.ndim != 2 or measured.shape != trace.shape:
        raise ValueError("measured and trace must have matching [view, detector] shapes")
    if endpoint.device != measured.device or endpoint.device != trace.device:
        raise ValueError("endpoint, measured and trace must be on the same device")
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    if not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be between zero and one")
    if trace.dtype != torch.bool and not ((trace == 0) | (trace == 1)).all():
        raise ValueError("trace must contain only zero and one")
    reliable = trace == 0
    if not (endpoint.isfinite().all() & (measured.isfinite() | ~reliable).all()):
        raise ValueError("endpoint and reliable measurements must be finite")
    if not reliable.any():
        return endpoint.clone()

    projection_dtype = torch.float64 if endpoint.dtype == torch.float64 else torch.float32

    def forward(image):
        out = projector.forward(image.to(projection_dtype))
        if out.shape != measured.shape or out.device != endpoint.device:
            raise ValueError("projector.forward returned an incompatible sinogram")
        return out.double()

    def adjoint(sinogram):
        out = projector.adjoint(sinogram.to(projection_dtype))
        if out.shape != endpoint.shape or out.device != endpoint.device:
            raise ValueError("projector.adjoint returned an incompatible image")
        return out.double()

    x = endpoint.double().clone()
    residual = adjoint(torch.where(reliable, measured - forward(x), 0.0))
    direction = residual
    norm_sq = residual.square().sum()
    if not norm_sq.isfinite():
        raise ArithmeticError("nonfinite initial residual")
    target_sq = relative_tolerance ** 2 * norm_sq
    for _ in range(max_iterations):
        if norm_sq <= target_sq:
            break
        h_direction = adjoint(torch.where(reliable, forward(direction), 0.0)) + rho * direction
        curvature = (direction * h_direction).sum()
        if not (curvature.isfinite() & (curvature > 0)):
            raise ArithmeticError("CG lost positive curvature")
        step = norm_sq / curvature
        x = x + step * direction
        residual = residual - step * h_direction
        next_norm_sq = residual.square().sum()
        direction = residual + (next_norm_sq / norm_sq) * direction
        norm_sq = next_norm_sq
    result = x.to(endpoint.dtype)
    if not (norm_sq.isfinite() & result.isfinite().all()):
        raise ArithmeticError("nonfinite PCC solution")
    return result
