from typing import Protocol

import numpy as np


class ProjectionPair(Protocol):
    def forward(self, image: np.ndarray) -> np.ndarray: ...

    def adjoint(self, sinogram: np.ndarray) -> np.ndarray: ...


def solve_pcc(endpoint: np.ndarray, measured: np.ndarray, trace: np.ndarray,
              projector: ProjectionPair, *, rho: float = 1.0,
              max_iterations: int = 10, relative_tolerance: float = 1e-4) -> np.ndarray:
    endpoint = np.asarray(endpoint)
    measured = np.asarray(measured)
    trace = np.asarray(trace)
    if endpoint.ndim != 2 or not np.issubdtype(endpoint.dtype, np.floating):
        raise ValueError("endpoint must be a 2D floating-point attenuation image")
    if measured.ndim != 2 or trace.shape != measured.shape:
        raise ValueError("measured and trace must have matching [view, detector] shapes")
    if not np.all((trace == 0) | (trace == 1)):
        raise ValueError("trace must contain only zero and one")
    reliable = trace == 0
    if not np.isfinite(endpoint).all() or not np.isfinite(measured[reliable]).all():
        raise ValueError("endpoint and measurements outside the metal trace must be finite")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be between zero and one")

    def forward(value):
        out = np.asarray(projector.forward(value), dtype=np.float64)
        if out.shape != measured.shape or not np.isfinite(out).all():
            raise ValueError("projector.forward returned an invalid sinogram")
        return out

    def adjoint(value):
        out = np.asarray(projector.adjoint(value), dtype=np.float64)
        if out.shape != endpoint.shape or not np.isfinite(out).all():
            raise ValueError("projector.adjoint returned an invalid image")
        return out

    x = endpoint.astype(np.float64, copy=True)
    if not reliable.any():
        return endpoint.copy()
    discrepancy = np.zeros(measured.shape, dtype=np.float64)
    np.subtract(measured, forward(x), out=discrepancy, where=reliable)
    residual = adjoint(discrepancy)
    direction = residual.copy()
    norm_sq = np.sum(residual * residual)
    target_sq = relative_tolerance ** 2 * norm_sq
    for _ in range(max_iterations):
        if norm_sq <= target_sq:
            break
        projected = np.where(reliable, forward(direction), 0.0)
        h_direction = adjoint(projected) + rho * direction
        curvature = np.sum(direction * h_direction)
        if not np.isfinite(curvature) or curvature <= 0:
            raise ArithmeticError("CG lost positive curvature")
        step = norm_sq / curvature
        x += step * direction
        residual = residual - step * h_direction
        next_norm_sq = np.sum(residual * residual)
        direction = residual + (next_norm_sq / norm_sq) * direction
        norm_sq = next_norm_sq
    return x.astype(endpoint.dtype, copy=False)
