"""Portable data preparation and experiment execution."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

import torch
from safetensors.torch import load_file, save_file

from .experiments import configurations, reference_kind
from .modelj_data import ModelJDataset, load_reference
from .models import build_model
from .sd_data import discover_sd_classes, factor_dataset, load_sd_dataset, load_sd_reference
from .training import fit, load_bank, make_sd_loader, move_batch, seed_everything, training_mean


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _reference_path(root, model, layer):
    if root is None:
        raise ValueError("Pretrained-reference experiments require --reference-root")
    root = Path(root)
    candidates = [root / model / f"layer_{layer}.safetensors",
                  root / model / f"layer_{layer}" / "base_reference.safetensors",
                  root / f"{model}.safetensors", root / f"{model}.bin",
                  root / model / "model.safetensors", root / model / "pytorch_model.bin"]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No reference for {model} L{layer} under the supplied reference root")
    return existing[0]


def _manifest(datasets, data_root, banks=None):
    """Fingerprint selected tensor contents/labels, not absolute source paths."""
    root = Path(data_root).resolve()
    result = {}
    seen = set()
    for split, dataset in datasets.items():
        names = []
        for path in dataset.paths:
            lexical = Path(path).absolute()
            resolved = lexical.resolve()
            if resolved in seen:
                raise ValueError("A checkpoint is reused across dataset splits")
            seen.add(resolved)
            try:
                name = str(lexical.relative_to(root))
            except ValueError:
                # SD model files may be symlinks into an external cache.
                name = f"{split}/{path.parent.name}/{path.name}"
            names.append((name, path.stat().st_size))
        result[split] = {
            "count": len(dataset),
            "ordered_source_manifest_sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest(),
        }
        digest = getattr(dataset, "_selected_content_sha256", None)
        if digest is None:
            hasher = hashlib.sha256()
            if banks is not None:
                tensors = banks[split]
                for tensor in tensors:
                    hasher.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
                    hasher.update(memoryview(tensor.contiguous().numpy()).cast("B"))
            else:
                for index in range(len(dataset)):
                    for tensor in dataset.factors(index):
                        hasher.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
                        hasher.update(memoryview(tensor.contiguous().numpy()).cast("B"))
            digest = hasher.hexdigest()
            dataset._selected_content_sha256 = digest
        result[split]["selected_tensors_and_labels_sha256"] = digest
    return result


def load_modelj_inputs(job, data_root):
    datasets = {split: ModelJDataset(data_root, job.model, job.layer, split)
                for split in ("train", "val", "test")}
    train_set = datasets["train"]
    if any(item.input_shape != train_set.input_shape for item in datasets.values()):
        raise ValueError("Layer shapes differ across splits")
    banks = {split: load_bank(dataset) for split, dataset in datasets.items()}
    return datasets, banks


def load_sd_inputs(job, data_root, text_root):
    if text_root is None:
        raise ValueError("SD-LoRA needs --text-root; run prepare-text first")
    seen, heldout = discover_sd_classes(data_root, job.model)
    folder = Path(text_root) / job.model
    with (folder / "classes.json").open(encoding="utf-8") as stream:
        expected = json.load(stream)
    if expected.get("train") != seen or expected.get("holdout") != heldout:
        raise ValueError("Text feature class order does not match the local dataset")
    tensors = load_file(str(folder / "text_embeddings.safetensors"), device="cpu")
    for key, classes in (("train", seen), ("holdout", heldout)):
        if key not in tensors or tensors[key].shape != (len(classes), 768):
            raise ValueError("Invalid text feature dimensions")
        if not torch.isfinite(tensors[key]).all():
            raise ValueError("Nonfinite text features")
    datasets = {split: load_sd_dataset(data_root, job.model, job.layer, split,
                heldout if split.endswith("holdout") else seen)
                for split in ("train", "val", "test", "val_holdout", "test_holdout")}
    shape = datasets["train"].input_shape
    pair = (datasets["train"].layer_name_down, datasets["train"].layer_name_up)
    if any(dataset.input_shape != shape or
           (dataset.layer_name_down, dataset.layer_name_up) != pair
           for dataset in datasets.values()):
        raise ValueError("Selected SD-LoRA factor pair or matrix shape differs across splits")
    return datasets, {"id": tensors["train"], "zero_shot": tensors["holdout"]}


@torch.inference_mode()
def _sd_mean(dataset, config, device):
    total = torch.zeros(dataset.input_shape, dtype=torch.float64, device=device)
    count = 0
    loader = make_sd_loader(factor_dataset(dataset), config, 0, False)
    for batch in loader:
        weights, _ = move_batch(batch, device)
        total += weights.sum(0, dtype=torch.float64)
        count += len(weights)
    return (total / count).float().cpu()


def _random_reference(shape, job, device, match_norm_to=None):
    """A deterministic random reference, optionally rescaled to a target norm.

    The draw depends on the model, layer and seed, so each seed sees its own
    reference while the run stays reproducible.
    """
    key = f"{job.model}|{job.layer}|{job.seed}".encode()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63))
    matrix = torch.randn(*shape, generator=generator, dtype=torch.float32)
    if match_norm_to is not None:
        matrix = matrix * (match_norm_to.norm() / matrix.norm().clamp_min(1e-30))
    return matrix.to(device)


def _build_reference(kind, job, exemplar, banks, reference_root, train_config, device, *, contract):
    """Resolve the matrix a method subtracts, before any training of it."""
    if kind in {"pretrained", "random_norm"}:
        path = _reference_path(reference_root, job.model, job.layer)
        if job.task == "sd_lora":
            pretrained = load_sd_reference(path, exemplar.layer_name, exemplar.input_shape)
        else:
            pretrained = load_reference(path, job.model, exemplar.layer_name, exemplar.input_shape)
        if kind == "pretrained":
            return pretrained
        contract["random_norm_target_sha256"] = hashlib.sha256(
            pretrained.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        return _random_reference(exemplar.input_shape, job, device, match_norm_to=pretrained)
    if kind == "random":
        return _random_reference(exemplar.input_shape, job, device)
    if kind == "mean":
        return (training_mean(banks["train"], device) if job.task == "membership"
                else _sd_mean(exemplar, train_config, device))
    raise ValueError(f"unsupported reference kind: {kind!r}")


def run_job(job, *, data_root, reference_root, text_root, output, device,
            cap=1e-3, save_weights=False,
            learned_init="mean",
            resume=False, num_workers=2, epochs=None, prepared=None,
            n_probes=None, proj_dim=None, rep_dim=None, batch_size=None,
            learning_rate=None, weight_decay=None, eval_every=None):
    destination = Path(output) / job.key
    if job.task == "membership":
        datasets, banks = prepared if prepared is not None else load_modelj_inputs(job, data_root)
        texts = None
    else:
        datasets, texts = prepared if prepared is not None else load_sd_inputs(job, data_root, text_root)
        banks = None          # SD-LoRA references come from the dataset, not a bank
    exemplar = datasets["train"]
    model_config, train_config = configurations(job, exemplar.input_shape, cap=cap,
        n_probes=n_probes,
        proj_dim=proj_dim, rep_dim=rep_dim, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        eval_every=eval_every, epochs=epochs)
    train_config = replace(train_config, num_workers=num_workers)
    contract = {"job": asdict(job), "model": asdict(model_config),
                "training": asdict(train_config),
                "sources": _manifest(datasets, data_root, banks if job.task == "membership" else None),
                "layer_name": exemplar.layer_name}
    contract["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(Path(__file__).parent.glob("*.py"))}
    contract["torch_version"] = str(torch.__version__)
    contract["device_type"] = torch.device(device).type
    contract["numerics"] = {
        "num_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    reference_type = reference_kind(job.method)
    reference = None
    if reference_type == "learned":
        # The learned reference starts somewhere and is then trained; the
        # starting point is the subject of the initialisation ablation.
        reference = _build_reference(learned_init, job, exemplar, banks,
                                     reference_root, train_config, device, contract=contract)
        contract["learned_reference_init"] = learned_init
    elif reference_type != "none":
        reference = _build_reference(reference_type, job, exemplar, banks,
                                     reference_root, train_config, device, contract=contract)
    if reference is not None:
        contract["initial_reference_sha256"] = hashlib.sha256(
            reference.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    if texts is not None:
        contract["text_feature_sha256"] = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                                          for key, value in texts.items()}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    result_path = destination / "result.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Completed job already exists: {job.key}; use a new output or --resume")
        with result_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("contract_sha256") != fingerprint:
            raise ValueError("Cannot resume a result with a different data/configuration contract")
        if save_weights:
            groups = ("id",) if job.task == "membership" else ("id", "zero_shot")
            missing = [f"best_{group}.safetensors" for group in groups
                       if not (destination / f"best_{group}.safetensors").is_file()]
            if missing:
                raise FileNotFoundError(
                    "Completed results lack requested weight files: " + ", ".join(missing)
                    + ". Resume skips completed training and cannot recover unsaved states; "
                    "rerun with --save-weights in a fresh output directory."
                )
        print(f"completed: {job.key}", flush=True)
        return existing
    seed_everything(job.seed)
    if job.task == "membership":
        train = banks["train"]
        validation, tests = {"id": banks["val"]}, {"id": banks["test"]}
    else:
        loaders = {split: make_sd_loader(factor_dataset(dataset), train_config, job.seed,
                                         split == "train") for split, dataset in datasets.items()}
        train = loaders["train"]
        validation = {"id": loaders["val"], "zero_shot": loaders["val_holdout"]}
        tests = {"id": loaders["test"], "zero_shot": loaders["test_holdout"]}
    model = build_model(model_config, reference=reference)
    result, states = fit(model, train, validation, tests, train_config, job.seed, device, texts)
    result.update(contract=contract, contract_sha256=fingerprint, **asdict(job))
    if save_weights:
        destination.mkdir(parents=True, exist_ok=True)
        for group, state in states.items():
            save_file({key: value.contiguous() for key, value in state.items()},
                      str(destination / f"best_{group}.safetensors"))
    write_json(result_path, result)
    return result
