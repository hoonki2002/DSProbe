"""Shared local-bank and result utilities for the shared/offset protocols.

Only requested weight matrices and membership labels are loaded. No data are
downloaded, and no trained weights or intermediate features are saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from safetensors import safe_open

from .cap import ReferenceStepCap
from .modelj_data import MODELS, _membership_labels, matrix_from_tensor, resolve_layer_name


DEFAULT_LAYERS = {"ResNet": 55, "SupViT": 59, "MAE": 65, "DINO": 47}


def parse_args(experiment: str, variants: tuple[str, ...], argv=None):
    """Read a portable configuration, with explicitly supplied CLI overrides."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    first, _ = preliminary.parse_known_args(argv)
    defaults = {
        "data_root": "data", "output": f"results/{experiment}",
        "models": list(MODELS), "layers": dict(DEFAULT_LAYERS),
        "seeds": [1, 2, 3, 4, 5], "variants": list(variants),
        "n_probes": 128, "proj_dim": 128, "rep_dim": 512,
        "epochs": 500, "eval_every": 25, "batch_size": 128,
        "eval_batch_size": 128, "per_tree_batch": 32,
        "lr": 3e-4, "weight_decay": 1e-5, "cap": 0.0, "reference_root": "references",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "threads": 2,
    }
    if first.config:
        provided = json.loads(first.config.read_text(encoding="utf-8"))
        if not isinstance(provided, dict):
            raise ValueError("Configuration must be a JSON object")
        if provided.pop("experiment", experiment) != experiment:
            raise ValueError("Configuration belongs to a different experiment")
        unknown = set(provided) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        defaults.update(provided)
    parser = argparse.ArgumentParser(
        description=f"Run the {experiment} Model-J protocol using a local checkpoint bank.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-root", type=Path,
                        help="Pretrained roots (*_dsprobe variants only): references/MODEL.safetensors|.bin")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--layers", nargs="+", type=int,
                        help="Layer indices aligned with --models; otherwise use configured mapping")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=variants)
    for key in ("n_probes", "proj_dim", "rep_dim", "epochs", "eval_every", "batch_size",
                "eval_batch_size", "per_tree_batch", "threads"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--cap", type=float,
                        help="Optional per-step relative cap for learned references only; 0 disables")
    parser.add_argument("--device", help="e.g. cuda:0 or cpu; default auto-detects")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without loading data")
    parser.add_argument("--summarize", action="store_true", help="Aggregate completed JSON results only")
    parser.add_argument("--resume", action="store_true", help="Skip a complete result with identical inputs")
    parsed = parser.parse_args(argv)
    for key in defaults:
        value = getattr(parsed, key, None)
        if value is not None:
            defaults[key] = value
    for key, value in defaults.items():
        setattr(parsed, key, value)
    parsed.data_root, parsed.output = Path(parsed.data_root), Path(parsed.output)
    parsed.reference_root = Path(parsed.reference_root)
    if isinstance(parsed.layers, dict):
        missing = set(parsed.models) - set(parsed.layers)
        if missing:
            raise ValueError(f"Missing layer indices for {sorted(missing)}")
        parsed.layers = {name: parsed.layers[name] for name in parsed.models}
    else:
        if len(parsed.layers) != len(parsed.models):
            raise ValueError("--layers must have one index per selected model")
        parsed.layers = dict(zip(parsed.models, parsed.layers))
    for key in ("models", "seeds", "variants"):
        values = getattr(parsed, key)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be nonempty and contain no duplicates")
    if any(name not in MODELS for name in parsed.models):
        raise ValueError("Unknown architecture in configuration")
    if any(name not in variants for name in parsed.variants):
        raise ValueError("Unknown variant in configuration")
    for key in ("n_probes", "proj_dim", "rep_dim", "epochs", "eval_every", "batch_size",
                "eval_batch_size", "per_tree_batch", "threads"):
        value = getattr(parsed, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in parsed.seeds):
        raise ValueError("Seeds must be nonnegative integers")
    for key in ("lr", "weight_decay", "cap"):
        value = getattr(parsed, key)
        if not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            raise ValueError(f"Invalid {key}")
    for name, layer in parsed.layers.items():
        resolve_layer_name(name, layer)
    parsed.experiment = experiment
    return parsed


def protocol(args, variant: str) -> dict:
    keys = ("experiment", "models", "layers", "n_probes", "proj_dim", "rep_dim",
            "epochs", "eval_every", "batch_size", "eval_batch_size", "per_tree_batch",
            "lr", "weight_decay", "cap")
    result = {key: getattr(args, key) for key in keys}
    result.update(variant=variant, loss="BCEWithLogitsLoss", optimizer="Adam",
                  response_standardization="sample_std_over_matrix_plus_1e-8",
                  selection="validation_accuracy_strict_improvement",
                  device_type=torch.device(args.device).type)
    learned = variant in ("specialist", "shared_ref", "learned_reference")
    result["reference_initialization"] = ("float32_training_mean" if learned else
                                          "pretrained_fixed" if variant.endswith("_dsprobe") else "none")
    result["offset_initialization"] = ("negative_response_of_float32_training_mean"
                                        if variant == "mean_response" else
                                        "zero" if variant == "zero" else "none")
    if args.experiment == "shared" and not variant.startswith("specialist"):
        result["batch_schedule"] = "floor(max_tree_train_size/per_tree_batch), summed_tree_losses"
    return result


def code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("common.py", "shared.py", "offset.py", "modelj_data.py", "cap.py"):
        path = Path(__file__).parent / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_bank(root: Path, models, layers, device):
    """Load the released snapshot layout; reject duplicated leaves across splits.

    ``MODEL/SPLIT/*/snapshots/*/model.safetensors`` is supported both with and
    without Hugging Face refs files. Multiple revisions of one leaf are
    rejected rather than being counted as separate trained models.
    """
    data, shapes, fingerprints = {}, {}, {}
    for name in models:
        data[name], fingerprints[name] = {}, {}
        seen_ids, seen_paths = set(), set()
        shape = None
        key = resolve_layer_name(name, layers[name])
        for split in ("train", "val", "test"):
            paths = sorted((root / name / split).glob("*/snapshots/*/model.safetensors"))
            if not paths:
                raise FileNotFoundError(f"No Model-J checkpoints for {name}/{split} under {root}")
            matrices, labels = [], []
            digest = hashlib.sha256()
            for path in paths:
                identity = path.parents[2].name
                resolved = path.resolve()
                if identity in seen_ids or resolved in seen_paths:
                    raise ValueError(f"Repeated checkpoint identity in {name}: {identity}")
                seen_ids.add(identity)
                seen_paths.add(resolved)
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if key not in handle.keys():
                        raise KeyError(f"Selected layer {key} is missing from {name}/{split}/{identity}")
                    matrix = matrix_from_tensor(handle.get_tensor(key), name)
                    target = _membership_labels(handle.metadata())
                if shape is None:
                    shape = tuple(matrix.shape)
                if tuple(matrix.shape) != shape:
                    raise ValueError(f"Inconsistent matrix shape for {name}: {tuple(matrix.shape)} != {shape}")
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(matrix.numpy().tobytes())
                digest.update(target.numpy().tobytes())
                matrices.append(matrix)
                labels.append(target)
            data[name][split] = (torch.stack(matrices).to(device), torch.stack(labels).to(device))
            fingerprints[name][split] = {"count": len(paths), "sha256": digest.hexdigest(), "key": key}
        shapes[name] = shape
    return data, shapes, fingerprints


def batches(x, y, batch_size: int, shuffle: bool):
    order = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for start in range(0, len(x), batch_size):
        indices = order[start:start + batch_size]
        yield x[indices], y[indices]


@torch.no_grad()
def accuracy(forward: Callable, x, y, batch_size: int = 128) -> float:
    correct = total = 0
    for xb, yb in batches(x, y, batch_size, False):
        logits = forward(xb)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / total


def reference_caps(model, cap: float):
    if cap == 0:
        return []
    return [ReferenceStepCap(p, p.detach().clone(), cap)
            for name, p in model.named_parameters() if name.split(".")[-1] == "W_star"]


def update(optimizer, loss, caps):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    before = [cap.capture() for cap in caps]
    optimizer.step()
    for cap, previous in zip(caps, before):
        cap.apply(previous)


def fit_independent(model, data, args):
    """ZIP minibatch/selection protocol; best weights live in CPU RAM only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    caps = reference_caps(model, args.cap)
    best, selected_epoch, state, history = -1.0, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in batches(*data["train"], args.batch_size, True):
            optimizer.zero_grad(set_to_none=True)
            update(optimizer, criterion(model(xb), yb), caps)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            value = accuracy(model, *data["val"], args.eval_batch_size)
            history.append({"epoch": epoch, "validation_accuracy": value})
            if value > best:
                best, selected_epoch = value, epoch
                state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return {"test_accuracy": accuracy(model, *data["test"], args.eval_batch_size),
            "validation_accuracy": best, "selected_epoch": selected_epoch,
            "parameters": sum(p.numel() for p in model.parameters()), "history": history}


def result_path(args, variant: str, seed: int) -> Path:
    return args.output / variant / f"seed_{seed}.json"


def existing_result(path: Path, contract: dict, resume: bool):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"Result exists: {path}; use --resume for an identical run or a new output directory")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("contract") != contract or result.get("complete") is not True:
        raise ValueError(f"Existing result does not match the requested protocol/data/code: {path}")
    return result


def save_result(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(args):
    """Population SD across seeds; macro is formed within each seed first."""
    rows, contracts = [], {}
    for variant in args.variants:
        results = []
        for seed in args.seeds:
            path = result_path(args, variant, seed)
            if not path.is_file():
                raise FileNotFoundError(f"Missing requested seed result: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            contract = result.get("contract", {})
            observed = dict(contract.get("protocol", {}))
            expected = protocol(args, variant)
            # Summarizing a GPU run on a CPU-only host must not require a GPU.
            observed.pop("device_type", None)
            expected.pop("device_type", None)
            if (not result.get("complete") or contract.get("seed") != seed
                    or observed != expected):
                raise ValueError(f"Result does not match requested settings: {path}")
            invariant = {key: value for key, value in contract.items() if key != "seed"}
            if variant in contracts and contracts[variant] != invariant:
                raise ValueError(f"Mixed configurations, code versions, or data in {variant}")
            contracts[variant] = invariant
            results.append(result)
        for name in list(args.models) + ["Macro"]:
            values = [r["macro_accuracy"] if name == "Macro" else r["test_accuracy"][name] for r in results]
            rows.append({"variant": variant, "model": name, "seeds": len(values),
                         "mean": float(np.mean(values)), "population_sd": float(np.std(values, ddof=0))})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Model | Accuracy (%) | Seeds |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['variant']} | {r['model']} | {r['mean']:.3f} ± {r['population_sd']:.3f} | {r['seeds']} |"
              for r in rows]
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def prepare(args):
    if args.dry_run:
        print(json.dumps({"data_root": str(args.data_root), "output": str(args.output),
                          "seeds": args.seeds,
                          "protocols": [protocol(args, v) for v in args.variants]}, indent=2))
        return False
    torch.set_num_threads(args.threads)
    if args.summarize:
        summarize(args)
        return False
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU run")
    return True
