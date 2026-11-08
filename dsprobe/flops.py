"""Forward matrix-multiplication FLOPs, including batch-shared references."""

from __future__ import annotations

from numbers import Integral


def forward_flops(
    rows: int,
    columns: int,
    probes: int,
    *,
    branches: int = 4,
    has_reference: bool = False,
    batch_size: int = 128,
    proj_dim: int = 128,
    rep_dim: int = 512,
    num_classes: int = 100,
) -> float:
    """Return per-example membership forward FLOPs, counting a MAC as two.

    The four-view model evaluates XU, X(X^T U), X^T U, and X^T(XU),
    followed by branch projections, the encoder, and classification head.
    When present, its six reference matrix products are evaluated once per
    batch and add 12 * rows * columns * probes / batch_size per example.
    One-branch accounting covers the reference-free ProbeX/XU model only.
    Normalization, elementwise operations, losses, and backward are excluded.
    """
    dimensions = {
        "rows": rows, "columns": columns, "probes": probes,
        "branches": branches, "batch_size": batch_size,
        "proj_dim": proj_dim, "rep_dim": rep_dim, "num_classes": num_classes,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if branches not in (1, 4):
        raise ValueError("FLOP accounting supports one or four branches")
    if not isinstance(has_reference, bool):
        raise ValueError("has_reference must be a boolean")
    if has_reference and branches != 4:
        raise ValueError("reference FLOP accounting requires four branches")

    if branches == 4:
        response = 6 * rows * columns * probes
        projection = 2 * (rows + columns) * proj_dim * probes
    else:
        response = rows * columns * probes
        projection = rows * proj_dim * probes
    total = 2 * (response + projection + branches * proj_dim * probes * rep_dim
                 + rep_dim * num_classes)
    if has_reference:
        total += 12 * rows * columns * probes / batch_size
    return total


__all__ = ["forward_flops"]
