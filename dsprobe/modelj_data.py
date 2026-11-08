"""Local Model-J checkpoint-bank loading and reference matricization.

The bank is read without modification. No download, split construction, or
checkpoint execution is performed. Layer numbers are the zero-based indices
in the supplied layer-name tables, not architecture forward-order positions.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Sequence
import warnings

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


MODELS = ("ResNet", "SupViT", "MAE", "DINO")
LAYER_DIRECTORY = Path(__file__).resolve().parent / "assets" / "layers"
CIFAR100_CLASSES = tuple(sorted((
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
    "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
    "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm",
)))
CLASS_TO_INDEX = {name: index for index, name in enumerate(CIFAR100_CLASSES)}


def resolve_layer_name(model: str, layer: int) -> str:
    """Resolve the historical Model-J layer-index convention."""
    if model not in MODELS:
        raise ValueError(f"Unknown architecture: {model}")
    filename = "resnet.csv" if model == "ResNet" else "vit.csv"
    with (LAYER_DIRECTORY / filename).open(newline="", encoding="utf-8") as handle:
        names = [row["0"] for row in csv.DictReader(handle)]
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < len(names):
        raise ValueError(f"Layer must be an integer in [0, {len(names) - 1}]")
    return names[layer]


def matrix_from_tensor(tensor: torch.Tensor, model: str) -> torch.Tensor:
    """Apply the historical weight-to-matrix rule, returning CPU float32.

    In particular, ResNet convolutions use reshape(-1, output_channels), not
    the conventional output_channels-by-flattened-input matricization.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown architecture: {model}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("A checkpoint weight must be a tensor")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    squeezed = tensor.squeeze()
    if squeezed.ndim == 4 and model == "ResNet":
        tensor = squeezed.reshape(-1, squeezed.shape[0]).squeeze()
    elif tensor.ndim > 2:
        tensor = squeezed
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2 or min(tensor.shape) < 1:
        raise ValueError(f"Weight is not a matrix after preprocessing: {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Weight contains non-finite values")
    return tensor.contiguous()


def _membership_labels(metadata: dict[str, str] | None) -> torch.Tensor:
    try:
        names = ast.literal_eval((metadata or {})["dataset_chosen_targets"])
    except (KeyError, ValueError, SyntaxError, TypeError) as error:
        raise ValueError("Missing or invalid dataset_chosen_targets metadata") from error
    if not isinstance(names, (list, tuple, set)) or len(names) != 50:
        raise ValueError("Expected exactly 50 CIFAR100 class names")
    if not all(isinstance(name, str) and name in CLASS_TO_INDEX for name in names):
        raise ValueError("Labels must be known CIFAR100 class names")
    if len(set(names)) != 50:
        raise ValueError("Membership classes must be unique")
    labels = torch.zeros(100, dtype=torch.float32)
    labels[[CLASS_TO_INDEX[name] for name in names]] = 1.0
    return labels


class ModelJDataset(Dataset):
    """One architecture, split, and layer of a local HF-format model bank.

    ``data_root/MODEL/SPLIT`` must contain the original cache directories:
    ``models--.../refs/main`` and ``snapshots/REV/model.safetensors``.
    Incomplete caches are skipped, as in the historical loader. Resolvable
    checkpoint paths are sorted before indexing, matching the trainers.
    """

    def __init__(self, data_root: Path, model: str, layer: int, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val, or test")
        self.model = model
        self.layer = layer
        self.split = split
        self.layer_name = resolve_layer_name(model, layer)
        root = Path(data_root).expanduser() / model / split
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint-bank split does not exist: {root}")
        paths = []
        skipped = 0
        for directory in root.iterdir():
            if not directory.is_dir() or directory.name.startswith((".DS_Store", ".locks")):
                continue
            reference = directory / "refs" / "main"
            if not reference.is_file():
                skipped += 1
                continue
            revision = reference.read_text(encoding="utf-8").strip()
            if not revision or revision in (".", "..") or "/" in revision or "\\" in revision:
                raise ValueError(f"Invalid cache revision in {reference}")
            checkpoint = directory / "snapshots" / revision / "model.safetensors"
            if not checkpoint.is_file():
                skipped += 1
                continue
            paths.append(checkpoint)
        self.paths = tuple(sorted(paths))
        if not self.paths:
            raise ValueError(f"No usable checkpoints in {root}")
        if skipped:
            warnings.warn(
                f"Skipped {skipped} incomplete cache directories in {model}/{split}",
                stacklevel=2,
            )
        first_matrix, _ = self[0]
        self.input_shape = tuple(first_matrix.shape)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with safe_open(str(self.paths[index]), framework="pt", device="cpu") as handle:
            if self.layer_name not in handle.keys():
                raise KeyError(f"Checkpoint lacks layer {self.layer_name}")
            matrix = matrix_from_tensor(handle.get_tensor(self.layer_name), self.model)
            labels = _membership_labels(handle.metadata())
        expected = getattr(self, "input_shape", tuple(matrix.shape))
        if tuple(matrix.shape) != expected:
            raise ValueError(f"Inconsistent matrix shape: {tuple(matrix.shape)} != {expected}")
        return matrix, labels


def _candidate_reference_keys(layer_name: str) -> list[str]:
    keys = [layer_name]
    for prefix in ("vit.", "resnet.", "model.", "module."):
        if layer_name.startswith(prefix):
            keys.append(layer_name[len(prefix):])
    keys.extend((f"model.{layer_name}", f"module.{layer_name}"))
    keys.extend(("reference", "x_star", "train_mean"))
    return list(dict.fromkeys(keys))


def load_reference(
    path: Path, model: str, layer_name: str, shape: Sequence[int]
) -> torch.Tensor:
    """Load one frozen reference, never downloading or running model code.

    Accepts an upstream safetensors/PyTorch state dict or a matrix artifact
    with one of the keys ``reference``, ``x_star``, or ``train_mean``. Public
    backbone prefix differences are handled explicitly. Ambiguous matches
    are rejected. PyTorch checkpoints are loaded with ``weights_only=True``.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates = _candidate_reference_keys(layer_name)
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            matches = [key for key in candidates if key in handle.keys()]
            if len(matches) != 1:
                raise ValueError(f"Expected one reference tensor, found keys {matches}")
            tensor = handle.get_tensor(matches[0])
            metadata = handle.metadata() or {}
        if "model" in metadata and metadata["model"] != model:
            raise ValueError("Reference artifact architecture does not match the bank")
        if "modelj_key" in metadata and metadata["modelj_key"] != layer_name:
            raise ValueError("Reference artifact layer does not match the bank")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError("Reference checkpoint must contain a tensor state dict")
        matches = [key for key in candidates if key in payload]
        if len(matches) != 1:
            raise ValueError(f"Expected one reference tensor, found keys {matches}")
        tensor = payload[matches[0]]
    reference = matrix_from_tensor(tensor, model)
    expected = tuple(int(value) for value in shape)
    if tuple(reference.shape) != expected:
        raise ValueError(f"Reference shape {tuple(reference.shape)} != bank shape {expected}")
    return reference
