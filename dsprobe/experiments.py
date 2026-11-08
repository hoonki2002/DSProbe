"""Explicit experiment grids and method-specific historical protocols."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json

from .models import ModelConfig
from .training import TrainConfig

MODELS = ("ResNet", "SupViT", "MAE", "DINO")
COUNTS = {"ResNet": 105, "SupViT": 74, "MAE": 74, "DINO": 74}
METHODS = {
    "probex", "probex4", "mvprobe", "dsprobe", "dsprobe_mean", "dsprobe_rand",
    "dsprobe_rand_norm", "r_dsprobe",
    "linear_raw", "linear_pretrained", "linear_learned",
    "gram_left_raw", "gram_left_pretrained", "gram_left_learned",
    "gram_right_raw", "gram_right_pretrained", "gram_right_learned",
    "poly6", "poly8", "duplicate6", "duplicate8",
}


@dataclass(frozen=True)
class Job:
    task: str
    model: str
    layer: int
    method: str
    seed: int

    @property
    def key(self):
        return f"{self.model}/layer_{self.layer}/{self.method}/seed_{self.seed}"


def read_suite(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        suite = json.load(stream)
    required = {"task", "models", "layers", "methods", "seeds"}
    if not required <= suite.keys():
        raise ValueError(f"Suite is missing: {sorted(required - suite.keys())}")
    if suite["task"] not in {"membership", "sd_lora"}:
        raise ValueError("Unknown task")
    if not set(suite["methods"]) <= METHODS:
        raise ValueError("Unknown method in suite")
    if suite["task"] == "sd_lora" and not set(suite["methods"]) <= {"probex", "probex4", "mvprobe", "dsprobe", "r_dsprobe"}:
        raise ValueError("The SD-LoRA experiment supports ProbeX, MVProbe, DSProbe and R-DSProbe")
    valid_models = {"SD_200", "SD_1k"} if suite["task"] == "sd_lora" else set(MODELS)
    if not set(suite["models"]) <= valid_models:
        raise ValueError("Unknown model for the suite task")
    if not set(suite["models"]) <= suite["layers"].keys():
        raise ValueError("Every suite model requires a layer configuration")
    for key in ("models", "methods", "seeds"):
        if not suite[key] or len(suite[key]) != len(set(suite[key])):
            raise ValueError(f"Empty or duplicate {key}")
    return suite


def jobs_from_suite(suite, *, models=None, methods=None, layers=None, seeds=None, model_layers=None):
    selected_models = suite["models"] if models is None else models
    selected_methods = suite["methods"] if methods is None else methods
    selected_seeds = suite["seeds"] if seeds is None else seeds
    if not set(selected_models) <= set(suite["models"]) or not set(selected_methods) <= set(suite["methods"]):
        raise ValueError("Requested model/method is not part of this suite")
    if model_layers is not None:
        if layers is not None:
            raise ValueError("Choose either layers or model_layers, not both")
        if not set(model_layers) <= set(suite["models"]):
            raise ValueError("Layer override names a model absent from this suite")
    for model in selected_models:
        for method in selected_methods:
            configured = suite.get("layer_overrides", {}).get(method, {}).get(model, suite["layers"][model])
            requested = configured if layers is None else layers
            if model_layers is not None:
                requested = model_layers.get(model, requested)
            if isinstance(requested, int) and not isinstance(requested, bool):
                requested = [requested]
            count = 128 if suite["task"] == "sd_lora" else COUNTS[model]
            layer_ids = range(count) if requested == "all" else requested
            if not isinstance(layer_ids, (range, list, tuple)) or not layer_ids:
                raise ValueError("Layers must be a nonempty sequence of indices or 'all'")
            if len(layer_ids) != len(set(layer_ids)):
                raise ValueError("Layer indices must be unique")
            for layer in layer_ids:
                if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 or layer >= count:
                    raise ValueError(f"Invalid layer: {model} {layer}")
                # The upstream classifier cannot serve as a compatible origin.
                if suite["task"] == "membership" and method == "dsprobe" and layer == 0:
                    continue
                for seed in selected_seeds:
                    if not isinstance(seed, int) or seed < 0:
                        raise ValueError("Seeds must be nonnegative integers")
                    yield Job(suite["task"], model, layer, method, seed)


def reference_kind(method):
    """Which reference a method subtracts, and whether it is trained.

    ``dsprobe_mean`` and ``dsprobe_rand``/``dsprobe_rand_norm`` are frozen
    references used by the reference-choice ablation: the leaf mean (centering)
    and a random matrix, either as drawn or rescaled to the norm of the
    pretrained origin.
    """
    if method == "dsprobe" or method.endswith("_pretrained"):
        return "pretrained"
    if method == "dsprobe_mean":
        return "mean"
    if method == "dsprobe_rand":
        return "random"
    if method == "dsprobe_rand_norm":
        return "random_norm"
    if method == "r_dsprobe" or method.endswith("_learned"):
        return "learned"
    return "none"


def configurations(job, shape, *, cap=1e-3,
                   n_probes=None, proj_dim=None, rep_dim=None, batch_size=None,
                   learning_rate=None, weight_decay=None, eval_every=None, epochs=None):
    """Construct defaults and apply explicit, recorded architecture/training overrides.

    ``n_probes`` is the base per-branch budget K. ``probex4`` uses four times
    this number in its single branch; all other methods use K per branch.
    """
    family, response, standardize, probe_init = job.method, "linear", True, "xavier"
    base_probes = 128 if n_probes is None else n_probes
    probes = 4 * base_probes if job.method == "probex4" else base_probes
    legacy = job.method in {"probex", "probex4"} or job.method.startswith("linear_")
    if job.method in {"dsprobe", "dsprobe_mean", "dsprobe_rand",
                      "dsprobe_rand_norm", "r_dsprobe"}:
        family = "gramdiff"
    elif job.method == "probex4":
        family = "probex"
    elif job.method.startswith("linear_"):
        family, standardize, probe_init = "single", False, "default"
    elif job.method.startswith("gram_"):
        family = "single"
        side = "left" if "_left_" in job.method else "right"
        response = f"gram_{side}" if job.method.endswith("_raw") else f"diff_{side}"
    train = TrainConfig(cap=cap)
    if legacy:
        train = TrainConfig(shuffle="legacy", evaluate_final_epoch=False, cap=cap)
    if job.task == "membership" and (job.method == "probex" or family == "single"):
        train = replace(train, epochs=200, learning_rate=2e-3, weight_decay=0.0)
    if job.task == "sd_lora":
        probe_init = "default"
        train = TrainConfig(
            epochs=500, learning_rate=3e-4, weight_decay=1e-5,
            eval_every=10, later_ties=True, task="sd_lora", shuffle="sd_loader", cap=cap,
        )
    train_overrides = {key: value for key, value in {
        "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "eval_every": eval_every, "epochs": epochs,
    }.items() if value is not None}
    train = replace(train, **train_overrides)
    model = ModelConfig(
        input_shape=tuple(shape), family=family, single_response=response,
        n_probes=probes, standardize=standardize, probe_init=probe_init,
        proj_dim=128 if proj_dim is None else proj_dim,
        rep_dim=512 if rep_dim is None else rep_dim,
        reference_trainable=reference_kind(job.method) == "learned", task=job.task,
    )
    return model, train
