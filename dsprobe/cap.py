"""Post-optimizer reference-step cap, leaving Adam moments untouched."""

from __future__ import annotations

import math

import torch


class ReferenceStepCap:
    """Bound each actual reference update relative to the initial norm.

    Call ``before = cap.capture()`` immediately before ``optimizer.step()``
    and ``cap.apply(before)`` immediately afterwards. A zero cap disables
    clipping. The denominator is fixed at initialization, not the current
    reference norm; no anchor penalty is added and no optimizer state changes.
    Float32 arithmetic matches the training protocol.
    """

    def __init__(self, x_star: torch.Tensor, initial_reference: torch.Tensor,
                 max_relative_step: float) -> None:
        if tuple(x_star.shape) != tuple(initial_reference.shape):
            raise ValueError("x_star and initial_reference shapes must match")
        if not x_star.requires_grad:
            raise ValueError("x_star must be trainable")
        if x_star.dtype != torch.float32:
            raise ValueError("the reference-step cap requires float32 x_star")
        if not math.isfinite(max_relative_step) or max_relative_step < 0:
            raise ValueError("max_relative_step must be finite and nonnegative")
        if not torch.isfinite(initial_reference).all():
            raise ValueError("initial_reference contains non-finite values")
        self.x_star = x_star
        self.max_relative_step = float(max_relative_step)
        initial = initial_reference.detach().to(device=x_star.device, dtype=torch.float32)
        self._initial_norm = initial.square().sum().clamp_min(1e-24).sqrt()

    def capture(self) -> torch.Tensor:
        """Snapshot the reference immediately before the optimizer update."""
        return self.x_star.detach().clone()

    @torch.no_grad()
    def apply(self, before: torch.Tensor) -> dict[str, float | bool]:
        """Apply the cap to the actual update and return step diagnostics."""
        if before.shape != self.x_star.shape or before.dtype != torch.float32:
            raise ValueError("before must be a float32 reference snapshot")
        raw_update = self.x_star - before
        initial_norm = self._initial_norm.to(self.x_star.device)
        raw_relative = float(torch.linalg.vector_norm(raw_update) / initial_norm)
        if not math.isfinite(raw_relative):
            raise FloatingPointError("non-finite reference update")
        clipped = self.max_relative_step > 0 and raw_relative > self.max_relative_step
        if clipped:
            scale = self.max_relative_step / max(raw_relative, 1e-30)
            self.x_star.copy_(before + raw_update * scale)
        applied_relative = float(torch.linalg.vector_norm(self.x_star - before) / initial_norm)
        return {"raw_relative_update": raw_relative,
                "applied_relative_update": applied_relative, "clipped": clipped}


__all__ = ["ReferenceStepCap"]
