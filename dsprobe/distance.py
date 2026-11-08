"""How far the learned reference travels from the pretrained weight.

The learned reference is initialized at the training-bank mean and then trained,
so the question the table asks is whether training pulls it toward the
pretrained origin or away from it. Both distances are relative, in percent:

    d_pre(W') = ||W' - W_pre||_F / ||W_pre||_F * 100

Run the training suite with ``--save-weights`` first; this reads the saved
``x_star`` from each seed's selected state and the pretrained reference from
``--reference-root``, and recomputes the initial mean from the training bank.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from .experiments import Job, reference_kind
from .modelj_data import load_reference
from .runner import _reference_path, load_modelj_inputs
from .training import training_mean


def relative_distance(matrix: torch.Tensor, pretrained: torch.Tensor) -> float:
    if matrix.shape != pretrained.shape:
        raise ValueError(f"shape mismatch: {tuple(matrix.shape)} vs {tuple(pretrained.shape)}")
    matrix = matrix.detach().to(torch.float64)
    pretrained = pretrained.detach().to(torch.float64)
    return float(torch.linalg.matrix_norm(matrix - pretrained)
                 / torch.linalg.matrix_norm(pretrained) * 100)


def _learned_references(results_root: Path):
    """Saved x_star per (model, layer, method, seed), from selected states only."""
    found = defaultdict(dict)
    for path in sorted(Path(results_root).rglob("result.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if reference_kind(result["method"]) != "learned":
            continue
        weights = path.parent / "best_id.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights} is missing; rerun that suite with --save-weights "
                "in a fresh output directory")
        state = load_file(str(weights))
        if "x_star" not in state:
            raise ValueError(f"{weights} holds no x_star")
        found[(result["model"], result["layer"], result["method"])][result["seed"]] = state["x_star"]
    if not found:
        raise SystemExit(f"No learned-reference results with saved weights under {results_root}")
    return found


def report_distances(results_root, data_root, reference_root, output) -> list[dict]:
    rows = []
    for (model, layer, method), by_seed in sorted(_learned_references(Path(results_root)).items()):
        job = Job(task="membership", model=model, layer=layer, method=method, seed=1)
        datasets, banks = load_modelj_inputs(job, data_root)
        shape = datasets["train"].input_shape
        pretrained = load_reference(_reference_path(reference_root, model, layer),
                                    model, datasets["train"].layer_name, shape)
        mean = training_mean(banks["train"])
        distances = [relative_distance(by_seed[seed], pretrained) for seed in sorted(by_seed)]
        rows.append({
            "model": model, "layer": layer, "method": method, "seeds": len(distances),
            "d_pre_mean_initialization": round(relative_distance(mean, pretrained), 3),
            "d_pre_learned_reference": round(statistics.fmean(distances), 3),
            "population_std": round(statistics.pstdev(distances), 3) if len(distances) > 1 else 0.0,
        })
        rows[-1]["change"] = round(rows[-1]["d_pre_learned_reference"]
                                   - rows[-1]["d_pre_mean_initialization"], 3)
        del datasets, banks
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "reference_distance.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"  {row['model']:>7} L{row['layer']:<3} "
              f"d_pre(W_mean)={row['d_pre_mean_initialization']:>6.3f}%  "
              f"d_pre(W*)={row['d_pre_learned_reference']:>6.3f}% "
              f"±{row['population_std']:<5.3f}  change={row['change']:+.3f}%  "
              f"({row['seeds']} seeds)")
    print(f"\n-> {destination}")
    return rows
