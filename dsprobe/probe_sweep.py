"""Accuracy against the number of probes, and the compute that buys.

Fixing r=128 isolates the input representation; this sweep asks the other
question -- how few probes a method needs. Each method probes its own
validation-selected layer (with ResNet held at L55 for both methods). The
summary reports forward matrix-multiplication FLOPs including reference
responses shared across each batch, using the saved architecture dimensions.

    python -m dsprobe.probe_sweep --probes 1,2,4,8,16,32,64,128 \\
        --data-root data --reference-root references --output results/sweep
    python -m dsprobe.probe_sweep --output results/sweep --summarize
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from itertools import groupby
from pathlib import Path

import torch

from .experiments import configurations, jobs_from_suite, read_suite
from .runner import load_modelj_inputs, run_job
from .flops import forward_flops

DEFAULT_PROBES = (1, 2, 4, 8, 16, 32, 64, 128)


def sweep(args) -> None:
    suite = read_suite(args.suite)
    device = args.device
    if not args.dry_run and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    for probes in args.probes:
        jobs = list(jobs_from_suite(suite, models=args.models, methods=args.methods,
                                    seeds=args.seeds))
        destination = Path(args.output) / f"probes_{probes}"
        if args.dry_run:
            for job in jobs:
                model, training = configurations(job, (2, 3), n_probes=probes)
                print(f"{destination.name}/{job.key}: K={model.n_probes}, epochs={training.epochs}")
            continue
        jobs.sort(key=lambda job: (job.task, job.model, job.layer, job.method, job.seed))
        for _, group in groupby(jobs, key=lambda job: (job.task, job.model, job.layer)):
            group = list(group)
            prepared = load_modelj_inputs(group[0], args.data_root)
            for job in group:
                print(f"running: probes={probes} {job.key}", flush=True)
                run_job(job, data_root=args.data_root, reference_root=args.reference_root,
                        text_root=None, output=destination, device=device,
                        resume=args.resume, prepared=prepared, n_probes=probes)
            del prepared
    if args.dry_run:
        print(f"{len(args.probes)} probe counts")


def _result_cost(result, batch_size_override=None) -> dict:
    """Count a saved membership architecture, including its reference products."""
    model = result["contract"]["model"]
    family = model["family"]
    if model["task"] != "membership" or family not in {"mvprobe", "gramdiff", "probex"}:
        raise ValueError("Probe-sweep FLOPs support membership ProbeX/MVProbe/Gram-difference models")
    rows, columns = model["input_shape"]
    batch_size = (result["contract"]["training"]["batch_size"]
                  if batch_size_override is None else batch_size_override)
    dimensions = {name: model[name] for name in ("proj_dim", "rep_dim", "num_classes")}
    return {
        "input_rows": rows, "input_columns": columns,
        "batch_size": batch_size, **dimensions,
        "forward_flops_per_example": forward_flops(
            rows, columns, model["n_probes"],
            branches=1 if family == "probex" else 4,
            has_reference=family == "gramdiff", batch_size=batch_size, **dimensions),
    }


def summarize(args) -> None:
    """Aggregate seeds with reference-inclusive FLOPs at a stated batch size."""
    rows = []
    for path in sorted(Path(args.output).rglob("result.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        accuracy = float(result["results"]["id"]["test"]["accuracy"])
        if not math.isfinite(accuracy):
            raise ValueError(f"Nonfinite test accuracy in {path}")
        rows.append({
            "model": result["model"], "method": result["method"], "layer": result["layer"],
            "probes": result["contract"]["model"]["n_probes"], "seed": result["seed"],
            "test_accuracy": accuracy, "parameters": result["parameters"],
            **_result_cost(result, getattr(args, "flops_batch_size", None)),
        })
    if not rows:
        raise SystemExit(f"No results under {args.output}")
    key = lambda row: (row["model"], row["method"], row["probes"])
    summary = []
    cost_fields = ("input_rows", "input_columns", "batch_size", "proj_dim", "rep_dim",
                   "num_classes", "forward_flops_per_example")
    for (model, method, probes), group in groupby(sorted(rows, key=key), key=key):
        group = list(group)
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError(f"Duplicate seed for {model}/{method}/r={probes}")
        compared = ("layer", "parameters", *cost_fields)
        if len({tuple(row[field] for field in compared) for row in group}) != 1:
            raise ValueError(f"Incompatible FLOP configurations for {model}/{method}/r={probes}")
        first = group[0]
        values = [row["test_accuracy"] for row in group]
        summary.append({
            "model": model, "method": method, "layer": first["layer"], "probes": probes,
            "seeds": len(values), "mean_test_accuracy": round(statistics.fmean(values), 3),
            "population_std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
            "parameters": first["parameters"],
            **{field: first[field] for field in cost_fields},
        })
    baselines = {row["model"]: row for row in summary
                 if row["method"] == "mvprobe" and row["probes"] == 128}
    compatible = ("input_rows", "input_columns", "batch_size", "proj_dim", "rep_dim", "num_classes")
    for row in summary:
        baseline = baselines.get(row["model"])
        comparable = baseline is not None and all(row[field] == baseline[field] for field in compatible)
        row["mvprobe_128_flops_per_example"] = baseline["forward_flops_per_example"] if comparable else ""
        row["reduction_vs_mvprobe_128"] = (
            baseline["forward_flops_per_example"] / row["forward_flops_per_example"] if comparable else "")
    output = Path(args.output) / "probe_sweep_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        ratio = row["reduction_vs_mvprobe_128"]
        reduction = f"  MVProbe r=128 / this run: {ratio:.2f}x" if ratio != "" else ""
        print(f"  {row['model']:>7} {row['method']:>10} L{row['layer']:<3} r={row['probes']:>4} "
              f"{row['mean_test_accuracy']:>7.2f} ±{row['population_std']:<5.2f} "
              f"{row['forward_flops_per_example']/1e6:>9.3f}M forward FLOPs "
              f"(reference included; B={row['batch_size']}; {row['seeds']} seeds){reduction}")
    print(f"\n-> {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", type=Path, default=Path("configs/probe_sweep.json"))
    parser.add_argument("--probes", type=lambda text: [int(v) for v in text.split(",")],
                        default=list(DEFAULT_PROBES))
    parser.add_argument("--models", type=lambda t: t.split(","))
    parser.add_argument("--methods", type=lambda t: t.split(","))
    parser.add_argument("--seeds", type=lambda t: [int(v) for v in t.split(",")])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reference-root", type=Path, default=Path("references"))
    parser.add_argument("--output", type=Path, default=Path("results/probe_sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--flops-batch-size", type=int,
                        help="Accounting batch size for --summarize; default: saved training batch size")
    args = parser.parse_args(argv)
    if args.flops_batch_size is not None and (args.flops_batch_size < 1 or not args.summarize):
        parser.error("--flops-batch-size must be positive and used with --summarize")
    if not args.probes or any(probes < 1 for probes in args.probes) or len(set(args.probes)) != len(args.probes):
        raise ValueError("Probe counts must be unique positive integers")
    summarize(args) if args.summarize else sweep(args)


if __name__ == "__main__":
    main()
