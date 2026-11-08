"""Fixed-layer parameter counts and symbolic matrix-multiplication costs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from .flops import forward_flops
from .models import ModelConfig, build_model


_BATCH_SIZE = 128
# Cost depends on the probed matrix only through its shape, so the accounting
# layer is chosen for shape, not for accuracy. SupViT and MAE are 768x768 at
# either method's selected layer (53/59 and 65/64), leaving the numbers
# unchanged. ResNet is the exception: the two selected layers differ in size
# (L98 is 2048x512, L55 is 1024x256), and a larger layer inflates the reference
# W* -- the quantity this table measures -- without either method gaining. We
# therefore report ResNet at L55, the same iso-cost choice the probe-count
# comparison makes.
_COST_LAYERS = (
    ("ResNet", 55, (1024, 256)),
    ("SupViT", 59, (768, 768)),
    ("MAE", 65, (768, 768)),
    ("DINO", 47, (768, 768)),
)
_VARIANTS = (
    ("mvprobe", "mvprobe"),
    ("gramdiff", "gramdiff"),
)
# These are the paper's supplied probe-count comparison settings. This report
# evaluates their cost; it does not select probe counts or establish accuracy.
_PROBE_EFFICIENCY = (
    ("ResNet", 55, 55, (1024, 256), 6),
    ("SupViT", 59, 53, (768, 768), 24),
    ("MAE", 64, 65, (768, 768), 32),
    ("DINO", 47, 47, (768, 768), 96),
)
_MATRIX_OPERATORS = {"aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm"}


def _matrix_counts(counter: FlopCounterMode) -> dict[str, int]:
    """Restrict the count to matrix products even if counter coverage grows."""
    return {
        str(operator): int(count)
        for operator, count in counter.get_flop_counts().get("Global", {}).items()
        if str(operator) in _MATRIX_OPERATORS
    }


def _measure(shape: tuple[int, int], family: str, batch_size: int = _BATCH_SIZE) -> dict:
    expected_forward = forward_flops(
        *shape, 128, has_reference=family == "gramdiff", batch_size=batch_size)
    config = ModelConfig(
        input_shape=shape, family=family, num_classes=100,
        n_probes=128, proj_dim=128, rep_dim=512,
        reference_trainable=family == "gramdiff",
    )
    # Validate a real finite reference before moving the model to meta. The
    # symbolic graph then allocates no weight-batch or gradient storage.
    reference = torch.zeros(shape) if family == "gramdiff" else None
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, reference=reference)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    reference_count = model.x_star.numel() if family == "gramdiff" else 0
    model = model.to("meta")
    weights = torch.empty(batch_size, *shape, device="meta", requires_grad=False)
    with FlopCounterMode(display=False) as forward:
        loss = model(weights).sum()
    with FlopCounterMode(display=False) as backward:
        loss.backward()
    forward_ops, backward_ops = _matrix_counts(forward), _matrix_counts(backward)
    forward_count, backward_count = sum(forward_ops.values()), sum(backward_ops.values())
    if forward_count <= 0 or backward_count <= 0:
        raise RuntimeError("matrix FLOP counting returned an empty graph")
    if not math.isclose(forward_count / batch_size, expected_forward, rel_tol=1e-12):
        raise RuntimeError("symbolic forward FLOPs disagree with analytic accounting")
    train_flops = forward_count + backward_count
    return {
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "reference_parameter_count": reference_count,
        "batch_size": batch_size,
        "n_probes_per_branch": 128,
        "branch_count": 4,
        "proj_dim": 128,
        "rep_dim": 512,
        "num_classes": 100,
        "forward_flops_batch": forward_count,
        "actual_backward_flops_batch": backward_count,
        "actual_train_flops_batch": train_flops,
        "forward_flops_per_example": forward_count / batch_size,
        "actual_backward_flops_per_example": backward_count / batch_size,
        "actual_train_flops_per_example": train_flops / batch_size,
        "estimated_3x_forward_flops_per_example": 3 * forward_count / batch_size,
        "forward_gflops_per_example": forward_count / batch_size / 1e9,
        "actual_backward_gflops_per_example": backward_count / batch_size / 1e9,
        "actual_train_gflops_per_example": train_flops / batch_size / 1e9,
        "estimated_3x_forward_gflops_per_example": 3 * forward_count / batch_size / 1e9,
        "forward_operator_flops_batch": forward_ops,
        "backward_operator_flops_batch": backward_ops,
    }


def _probe_efficiency_report(batch_size: int) -> dict:
    records = []
    for model, mv_layer, ds_layer, shape, probes in _PROBE_EFFICIENCY:
        mv_flops = forward_flops(*shape, 128, batch_size=batch_size)
        ds_flops = forward_flops(*shape, probes, has_reference=True, batch_size=batch_size)
        records.append({
            "model": model,
            "mvprobe_layer": mv_layer,
            "r_dsprobe_layer": ds_layer,
            "input_rows": shape[0],
            "input_columns": shape[1],
            "mvprobe_probes": 128,
            "r_dsprobe_probes": probes,
            "batch_size": batch_size,
            "mvprobe_flops_per_example": mv_flops,
            "r_dsprobe_flops_per_example": ds_flops,
            "reduction": mv_flops / ds_flops,
        })
    return {
        "scope": "per-example forward matrix-multiplication FLOPs including reference products",
        "multiply_add_flops": 2,
        "batch_size": batch_size,
        "branch_count": 4,
        "proj_dim": 128,
        "rep_dim": 512,
        "num_classes": 100,
        "reference_accounting": "six reference matrix products shared across the batch; "
                                "12*m*n*r/B FLOPs per example included in R-DSProbe",
        "comparison_inputs": "paper-specified layers and probe counts used only as accounting inputs; "
                             "no accuracy matching, selection, or retraining is performed",
        "reduction_definition": "MVProbe forward FLOPs divided by R-DSProbe forward FLOPs",
        "excluded_operations": [
            "backward", "optimizer updates", "reference-step cap", "normalization",
            "elementwise operations", "reductions", "loss computation",
        ],
        "records": records,
    }


def report_costs(output: str | Path, batch_size: int = _BATCH_SIZE) -> dict:
    """Write fixed-K costs and paper probe-count efficiency JSON/CSV reports.

    Actual backward costs follow the trainable-parameter autograd graph with
    frozen input checkpoint matrices. They are not inferred by multiplying
    forward costs. The conventional three-times-forward estimate is reported
    in separate columns. These are symbolic operation counts, not timings.
    """
    cache = {}
    records = []
    for model_name, layer, shape in _COST_LAYERS:
        for variant, family in _VARIANTS:
            key = (shape, variant)
            if key not in cache:
                cache[key] = _measure(shape, family, batch_size=batch_size)
            baseline = cache[(shape, "mvprobe")]
            measured = cache[key]
            records.append({
                "model": model_name,
                "layer": layer,
                "input_rows": shape[0],
                "input_columns": shape[1],
                "variant": variant,
                **measured,
                "extra_parameters_vs_mvprobe": measured["parameter_count"] - baseline["parameter_count"],
            })
    report = {
        "scope": "symbolic forward/backward matrix-multiplication FLOPs at fixed cost layers",
        "layer_selection": ("shape-matched: ResNet at L55 (1024x256), not R-DSProbe's L98, "
                            "so the reference overhead is not inflated by a larger matrix; "
                            "SupViT/MAE/DINO are 768x768 at either method's selected layer"),
        "multiply_add_flops": 2,
        "parameter_scope": "allocated parameters including the trainable reference; buffers excluded",
        "input_requires_grad": False,
        "backward_objective": "sum of classifier logits",
        "actual_train_definition": "counted forward plus counted autograd backward",
        "estimate_definition": "three times counted forward; not actual backward or training cost",
        "excluded_operations": [
            "Adam and optimizer-state updates", "reference-step cap", "normalization",
            "elementwise operations", "reductions", "loss computation",
        ],
        "gram_difference": "raw responses minus unbatched reference responses; "
                           "the reference products are shared across the batch",
        "records": records,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "costs.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Operator-level details remain in JSON; CSV contains one flat row per
    # architecture/variant for convenient plotting and table generation.
    csv_fields = [key for key in records[0] if not key.endswith("operator_flops_batch")]
    with (output / "costs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    efficiency = _probe_efficiency_report(batch_size)
    with (output / "probe_efficiency.json").open("w", encoding="utf-8") as stream:
        json.dump(efficiency, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "probe_efficiency.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(efficiency["records"][0]))
        writer.writeheader()
        writer.writerows(efficiency["records"])
    return report


__all__ = ["report_costs"]
