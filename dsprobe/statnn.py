"""Full-model StatNN statistics with the historical Model-J XGBoost adapter.

This is not a selected-layer neural probe or the original accuracy-regression
task. Public architecture manifests replace Transformers model construction;
they preserve the exact sorted named_parameters() used in the server runs.
Checkpoint metadata supplies targets only and never enters the features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
from safetensors import safe_open
import torch

from .modelj_data import CIFAR100_CLASSES, MODELS, _membership_labels
from .runner import write_json

METHOD = "statnn_fullmodel_xgboost"
SPLITS = ("train", "val", "test")
FEATURE_ORDER = ("q0", "q25", "q50", "q75", "q100", "std_ddof0", "mean")
SCHEMA_PATH = Path(__file__).parent / "assets" / "statnn_parameters.json"
DEFAULT_HYPERPARAMETERS = dict(
    n_estimators=500, early_stopping_rounds=25, learning_rate=0.05,
    max_depth=3, subsample=0.9, colsample_bytree=1.0, reg_lambda=1.0,
    reg_alpha=0.0, tree_method="hist", device="cpu", n_jobs=4,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parameter_schema(model):
    if model not in MODELS:
        raise ValueError(f"Unknown StatNN architecture: {model}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))[model]
    keys, shapes = schema["parameter_keys"], schema["shapes"]
    if not keys or keys != sorted(set(keys)) or set(keys) != set(shapes):
        raise ValueError("Invalid named-parameter manifest")
    return schema


def tensor_statistics(values):
    """Upstream order and NumPy precision: percentiles, population std, mean.

    Do not preconvert float32 inputs to float64: the historical std/mean
    reductions were float32 before appending to the float64 percentiles.
    """
    flat = np.asarray(values).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("Empty or nonfinite parameter tensor")
    quantiles = np.percentile(flat, (0, 25, 50, 75, 100))
    return np.concatenate((quantiles, np.asarray([np.std(flat, ddof=0), np.mean(flat)])))


def inventory(data_root, model):
    """Keep the historical resolved-path order, including HF blob symlinks."""
    if model not in MODELS:
        raise ValueError(f"Unknown architecture: {model}")
    result, seen = {}, set()
    for split in SPLITS:
        root = Path(data_root).expanduser() / model / split
        if not root.is_dir():
            raise FileNotFoundError(root)
        rows, skipped = [], 0
        for folder in root.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            ref = folder / "refs" / "main"
            if not ref.is_file():
                skipped += 1
                continue
            revision = ref.read_text(encoding="utf-8").strip()
            if not revision or revision in (".", "..") or "/" in revision or "\\" in revision:
                raise ValueError(f"Invalid cache revision: {ref}")
            path = folder / "snapshots" / revision / "model.safetensors"
            if not path.is_file():
                skipped += 1
                continue
            path = path.resolve()
            if str(path) in seen:
                raise ValueError("A checkpoint is reused within/across dataset splits")
            seen.add(str(path))
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("split") != split:
                    raise ValueError(f"Checkpoint split metadata does not match {split}: {path}")
                if "model_idx" not in metadata:
                    raise ValueError(f"Missing model_idx metadata: {path}")
                labels = _membership_labels(metadata).to(torch.uint8).tolist()
            stat = path.stat()
            rows.append(dict(path=str(path), model_idx=metadata["model_idx"],
                             labels=labels, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        if not rows:
            raise ValueError(f"No usable checkpoints in {root}")
        if skipped:
            warnings.warn(f"Skipped {skipped} incomplete caches in {model}/{split}", stacklevel=2)
        result[split] = sorted(rows, key=lambda row: row["path"])
    return result


def extract_checkpoint(row, schema):
    """Seven features per named parameter; buffers and metadata are excluded."""
    path = Path(row["path"])
    stat = path.stat()
    for key, actual in (("size_bytes", stat.st_size), ("mtime_ns", stat.st_mtime_ns)):
        if key in row and row[key] != actual:
            raise ValueError(f"Checkpoint changed since inventory: {path}")
    keys, shapes = schema["parameter_keys"], schema["shapes"]
    # Only BatchNorm buffers associated with a named affine parameter may be
    # present beyond the audited parameter set. Unknown weights are not ignored.
    buffers = {key[:-len("weight")] + suffix for key in keys
               if key.endswith("normalization.weight")
               for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        missing, unexpected = set(keys) - actual, actual - set(keys) - buffers
        if missing or unexpected:
            raise ValueError(f"Named-parameter schema mismatch: missing={sorted(missing)}, "
                             f"unexpected={sorted(unexpected)}")
        parts = []
        for key in keys:
            weight = handle.get_tensor(key)
            if list(weight.shape) != shapes[key]:
                raise ValueError(f"Wrong parameter shape for {key}: {list(weight.shape)} != {shapes[key]}")
            parts.append(tensor_statistics(weight.numpy()))
    features = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite extracted statistics")
    return features


def multilabel_metrics(probabilities, labels):
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Predictions must be finite and match the label shape")
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1 - 1e-12)
    target = labels.astype(np.float64)
    decisions = probabilities >= 0.5
    return dict(bit_accuracy=float(np.mean(decisions == labels)),
                binary_log_loss=float(-np.mean(target * np.log(clipped)
                                              + (1 - target) * np.log1p(-clipped))),
                exact_match_accuracy=float(np.mean(np.all(decisions == labels, axis=1))))


def _xgboost():
    try:
        import xgboost
        import sklearn  # Required by XGBClassifier, not by bare xgboost.
    except ImportError as error:
        raise RuntimeError("StatNN needs its optional dependency: pip install -e '.[statnn]'") from error
    if xgboost.__version__ != "3.2.0":
        raise RuntimeError("Historical StatNN reproduction requires xgboost==3.2.0")
    return xgboost


def validate_hyperparameters(hyperparameters):
    if set(hyperparameters) != set(DEFAULT_HYPERPARAMETERS):
        raise ValueError("StatNN hyperparameters must contain exactly the documented XGBoost settings")
    if hyperparameters["device"] != "cpu" or hyperparameters["tree_method"] != "hist":
        raise ValueError("Historical StatNN uses CPU histogram boosting")
    for key in ("n_estimators", "early_stopping_rounds", "max_depth", "n_jobs"):
        value = hyperparameters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"):
        value = hyperparameters[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid {key}")
        if value < 0 or (key in {"learning_rate", "subsample", "colsample_bytree"} and value == 0):
            raise ValueError(f"Invalid {key}")
        if key in {"subsample", "colsample_bytree"} and value > 1:
            raise ValueError(f"Invalid {key}")


def train_classifier(features, labels, seed, hyperparameters):
    """Validation logloss selects boosting length; test is predicted once."""
    validate_hyperparameters(hyperparameters)
    if set(features) != set(SPLITS) or set(labels) != set(SPLITS):
        raise ValueError("Expected train/val/test feature and label arrays")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    n_features = None
    for split in SPLITS:
        x, y = features[split], labels[split]
        if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError(f"Invalid {split} features")
        if (y.shape != (len(x), 100) or not np.isin(y, (0, 1)).all()
                or not np.all(y.sum(axis=1) == 50)):
            raise ValueError(f"Expected 100 binary targets with 50 positives for {split}")
        if n_features is not None and x.shape[1] != n_features:
            raise ValueError("Inconsistent feature dimensions across splits")
        n_features = x.shape[1]
    classifier = _xgboost().XGBClassifier(
        **hyperparameters, objective="binary:logistic", eval_metric="logloss",
        multi_strategy="one_output_per_tree", random_state=seed,
    )
    classifier.fit(features["train"], labels["train"],
                   eval_set=[(features["val"], labels["val"])], verbose=False)
    validation = classifier.predict_proba(features["val"])
    test = classifier.predict_proba(features["test"])
    return classifier, validation, test


def _arrays_digest(features, labels):
    digest = hashlib.sha256()
    for split in SPLITS:
        for values in (features[split], labels[split]):
            digest.update(str(values.shape).encode())
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


def _load_features(folder, rows, schema, workers):
    cache = folder / "features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            features = {s: stored[s + "_x"] for s in SPLITS}
            labels = {s: stored[s + "_y"] for s in SPLITS}
    else:
        features, labels = {}, {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for split in SPLITS:
                parts = []
                for index, vector in enumerate(pool.map(lambda row: extract_checkpoint(row, schema), rows[split])):
                    parts.append(vector)
                    if index % 50 == 0:
                        print(f"StatNN features {split}: {index + 1}/{len(rows[split])}", flush=True)
                features[split] = np.stack(parts)
                labels[split] = np.asarray([row["labels"] for row in rows[split]], dtype=np.uint8)
        staging = folder / "features.tmp.npz"
        np.savez_compressed(staging, **{s + "_x": features[s] for s in SPLITS},
                            **{s + "_y": labels[s] for s in SPLITS})
        staging.replace(cache)
    for split in SPLITS:
        expected_labels = np.asarray([row["labels"] for row in rows[split]], dtype=np.uint8)
        if (features[split].shape != (len(rows[split]), 7 * len(schema["parameter_keys"]))
                or features[split].dtype != np.float32 or not np.isfinite(features[split]).all()
                or labels[split].dtype != np.uint8 or not np.array_equal(labels[split], expected_labels)):
            raise ValueError(f"Invalid/corrupted StatNN feature cache: {split}")
    fingerprint = folder / "features.sha256.json"
    digest = _arrays_digest(features, labels)
    if fingerprint.exists():
        if json.loads(fingerprint.read_text()) != {"sha256": digest}:
            raise ValueError("StatNN feature cache fingerprint mismatch")
    else:
        write_json(fingerprint, {"sha256": digest})
    return features, labels, digest


def run_suite(args, suite):
    """Separate full-model runner; never construct layer jobs or an Adam model."""
    if (suite.get("task") != "membership" or suite.get("input_scope") != "all_named_parameters"
            or suite.get("methods") != ["statnn"] or "layers" in suite):
        raise ValueError("StatNN needs the full-model configs/statnn.json, without layers")
    for key in ("models", "seeds"):
        values = suite.get(key, [])
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Empty/duplicate StatNN {key}")
    models = suite["models"] if args.models is None else args.models
    seeds = suite["seeds"] if args.seeds is None else args.seeds
    if not set(suite["models"]) <= set(MODELS) or not models or not set(models) <= set(suite["models"]):
        raise ValueError("Unknown/unconfigured StatNN model")
    if len(models) != len(set(models)) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Empty/duplicate model or seed selection")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in [*suite["seeds"], *seeds]):
        raise ValueError("Seeds must be nonnegative integers")
    if args.methods is not None and args.methods != ["statnn"]:
        raise ValueError("StatNN suite supports only --methods statnn")
    for key in ("layers", "model_layers", "epochs", "n_probes", "proj_dim", "rep_dim",
                "batch_size", "learning_rate", "weight_decay", "eval_every", "reference_root", "text_root"):
        if getattr(args, key) is not None:
            raise ValueError(f"--{key.replace('_', '-')} is not a full-model StatNN option; edit suite hyperparameters")
    if args.cap is not None or args.learned_init is not None:
        raise ValueError("StatNN has no learned reference, cap, or initialization")
    if args.device not in (None, "cpu"):
        raise ValueError("Historical StatNN uses CPU XGBoost; omit --device or use --device cpu")
    if args.num_workers < 0 or args.threads < 1:
        raise ValueError("Invalid worker/thread count")
    hyperparameters = dict(suite.get("hyperparameters", DEFAULT_HYPERPARAMETERS))
    validate_hyperparameters(hyperparameters)
    if args.dry_run:
        for model in models:
            count = len(parameter_schema(model)["parameter_keys"])
            for seed in seeds:
                print(f"{model}/full_model/statnn/seed_{seed}: parameters={count} tensors, "
                      f"features={7 * count}, XGBoost CPU, "
                      f"rounds={hyperparameters['n_estimators']}, lr={hyperparameters['learning_rate']}, "
                      f"early_stopping={hyperparameters['early_stopping_rounds']} (validation logloss)")
        print(f"{len(models) * len(seeds)} runs")
        return
    xgb = _xgboost()
    torch.set_num_threads(args.threads)
    for model in models:
        rows, schema = inventory(args.data_root, model), parameter_schema(model)
        folder = args.output / model / "full_model" / "statnn"
        source_digest = hashlib.sha256(Path(__file__).read_bytes() + SCHEMA_PATH.read_bytes()).hexdigest()
        contract = dict(
            schema_version=1, method=METHOD, input_scope="all_named_parameters",
            model=dict(architecture=model, parameter_schema_sha256=_digest(schema),
                       feature_order=list(FEATURE_ORDER), n_features=7 * len(schema["parameter_keys"])),
            training=dict(hyperparameters=hyperparameters, objective="binary:logistic",
                          eval_metric="logloss", multi_strategy="one_output_per_tree",
                          selection="validation logloss early stopping", threshold=0.5),
            sources={s: dict(count=len(rows[s]), manifest_sha256=_digest(rows[s])) for s in SPLITS},
            source_sha256=source_digest, torch_version=str(torch.__version__), device_type="cpu",
            numerics=dict(numpy_version=np.__version__, xgboost_version=xgb.__version__,
                          sklearn_version=version("scikit-learn")),
            class_order=list(CIFAR100_CLASSES), layer_name="full_model",
        )
        contract_path = folder / "contract.json"
        if folder.exists():
            if not args.resume:
                raise FileExistsError(f"StatNN output exists: {folder}; use --resume or a new output")
            if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
                raise ValueError("StatNN resume contract mismatch (inputs, implementation or configuration changed)")
        else:
            folder.mkdir(parents=True, exist_ok=False)
            write_json(contract_path, contract)
        write_json(folder / "data_manifest.json", rows)
        features, labels, feature_digest = _load_features(folder, rows, schema, args.num_workers)
        result_contract = {**contract, "feature_sha256": feature_digest}
        for seed in seeds:
            target = folder / f"seed_{seed}"
            result_path = target / "result.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if (previous.get("contract") != result_contract or previous.get("seed") != seed
                        or previous.get("model") != model or previous.get("method") != METHOD
                        or not (target / "model.ubj").is_file() or not (target / "predictions.npz").is_file()):
                    raise ValueError("Invalid/incomplete StatNN result for resume")
                print(f"resume: {model}/full_model/statnn/seed_{seed}", flush=True)
                continue
            target.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            classifier, validation, test = train_classifier(features, labels, seed, hyperparameters)
            vm, tm = multilabel_metrics(validation, labels["val"]), multilabel_metrics(test, labels["test"])
            iteration = int(classifier.best_iteration)
            classifier.save_model(target / "model.ubj")
            np.savez_compressed(target / "predictions.npz", val=validation, test=test,
                                val_labels=labels["val"], test_labels=labels["test"])
            result = dict(task="membership", model=model, layer="full_model", method=METHOD, seed=seed,
                parameters=None, n_features=7 * len(schema["parameter_keys"]), contract=result_contract,
                results={"id": {
                    "validation": dict(accuracy=100 * vm["bit_accuracy"], loss=vm["binary_log_loss"], epoch=iteration),
                    "test": dict(accuracy=100 * tm["bit_accuracy"], loss=tm["binary_log_loss"]),
                }}, best_iteration=iteration, test_acc=tm["bit_accuracy"], val_acc=vm["bit_accuracy"],
                test_loss=tm["binary_log_loss"], val_loss=vm["binary_log_loss"],
                elapsed_seconds=time.monotonic() - start)
            write_json(result_path, result)
            print(f"{model}/full_model/statnn/seed_{seed}: test={100 * tm['bit_accuracy']:.3f}%, "
                  f"best_iteration={iteration}", flush=True)
        write_json(folder / "complete.json", dict(status="complete", requested_seeds=seeds, runs=len(seeds)))
