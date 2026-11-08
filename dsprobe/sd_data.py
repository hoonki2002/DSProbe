"""Local Model-J SD-LoRA data and explicitly requested CLIP text features.

Dataset construction never contacts a remote service. Text features use the
frozen OpenAI CLIP ViT-L/14 encoder; its learned projection belongs to the probe,
not to this module. Downloading CLIP weights requires ``allow_download=True``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test", "val_holdout", "test_holdout")
SUBSETS = ("SD_200", "SD_1k")
LORA_FILENAME = "pytorch_lora_weights.safetensors"


def _subset_root(root: Path, subset: str) -> Path:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    root = Path(root).expanduser()
    nested = root / subset
    if nested.is_dir():
        return nested
    # Also accept the subset directory itself for convenient local use.
    if root.is_dir() and (root / "train").is_dir():
        return root
    raise FileNotFoundError(f"No local {subset} dataset directory")


def _split_paths(root: Path, split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown SD-LoRA split: {split}")
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing local SD-LoRA split: {split}")
    paths: list[Path] = []
    visited: set[Path] = set()
    for directory, subdirs, filenames in os.walk(split_root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirs[:] = []
            continue
        visited.add(resolved)
        subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
        if LORA_FILENAME in filenames:
            paths.append(Path(directory) / LORA_FILENAME)
    # Match the original lexical source-path order before resolving symlinks.
    paths = [path.resolve() for path in sorted(paths)]
    if not paths:
        raise ValueError(f"No LoRA weight files in split: {split}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate resolved LoRA sources in split: {split}")
    return paths


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if not metadata.get("imagenet_class_id"):
        raise ValueError("A LoRA file is missing imagenet_class_id metadata")
    return metadata


def discover_sd_classes(root: Path, subset: str) -> tuple[list[str], list[str]]:
    """Return sorted training and held-out class IDs using local headers only.

    All five splits must be present. Validation/test sources must be disjoint,
    ID validation/test classes must occur in training, and held-out classes must
    not overlap training classes. No tensor payload is read by this function.
    """
    root = _subset_root(root, subset)
    paths = {split: _split_paths(root, split) for split in SPLITS}
    seen_sources: set[Path] = set()
    for split in SPLITS:
        if seen_sources.intersection(paths[split]):
            raise ValueError("LoRA source overlap between dataset splits")
        seen_sources.update(paths[split])
    ids = {
        split: {_metadata(path)["imagenet_class_id"] for path in paths[split]}
        for split in SPLITS
    }
    seen = ids["train"]
    heldout = ids["val_holdout"] | ids["test_holdout"]
    if seen & heldout:
        raise ValueError("Training and held-out class IDs overlap")
    for split in ("val", "test"):
        if ids[split] - seen:
            raise ValueError(f"{split} contains classes absent from training")
    return sorted(seen), sorted(heldout)


class SDLoRADataset(Dataset):
    """A single sorted split returning ``(B @ A, scalar torch.long label)``.

    ``classes`` is the sorted candidate-class vocabulary. ``paths`` contains
    resolved local source paths. ``metadata`` is the first sample's header,
    while ``sample_metadata`` contains every sample's header in source order.
    ``layer_name`` aliases ``layer_name_down``; both factor keys are exposed.
    """

    def __init__(self, root: Path, subset: str, layer: int, split: str,
                 classes: Sequence[str]) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        supplied = list(classes)
        if not supplied or any(not isinstance(cid, str) or not cid for cid in supplied):
            raise ValueError("classes must contain nonempty class-ID strings")
        if len(set(supplied)) != len(supplied):
            raise ValueError("Duplicate candidate class IDs")
        self.classes = sorted(supplied)
        self.subset = subset
        self.split = split
        self.layer = layer
        root = _subset_root(root, subset)
        self.paths = _split_paths(root, split)
        # The layer index is defined by the first training model, as in the
        # experiment, not by the first file of whichever split is requested.
        exemplar = _split_paths(root, "train")[0]
        with safe_open(str(exemplar), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            pairs = [
                (down, down.replace(".lora.down.weight", ".lora.up.weight"))
                for down in sorted(keys) if ".lora.down.weight" in down
                and down.replace(".lora.down.weight", ".lora.up.weight") in keys
            ]
        if layer >= len(pairs):
            raise IndexError(f"LoRA layer {layer} outside available range 0..{len(pairs) - 1}")
        self.layer_name_down, self.layer_name_up = pairs[layer]
        self.layer_name = self.layer_name_down
        self.layer_pair_count = len(pairs)
        class_to_index = {cid: index for index, cid in enumerate(self.classes)}
        self.sample_metadata: list[dict[str, str]] = []
        self._labels: list[int] = []
        self.input_shape: tuple[int, int] | None = None
        for path in self.paths:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
                keys = set(handle.keys())
                if not {self.layer_name_down, self.layer_name_up} <= keys:
                    raise ValueError("A LoRA file is missing the selected factor pair")
                a = tuple(handle.get_slice(self.layer_name_down).get_shape())
                b = tuple(handle.get_slice(self.layer_name_up).get_shape())
            if len(a) != 2 or len(b) != 2 or a[0] != b[1] or min(*a, *b) < 1:
                raise ValueError("Selected LoRA factors have incompatible shapes")
            shape = (b[0], a[1])
            if self.input_shape is not None and shape != self.input_shape:
                raise ValueError("Selected LoRA layer shapes differ between samples")
            self.input_shape = shape
            cid = metadata.get("imagenet_class_id")
            if cid not in class_to_index:
                raise ValueError("A LoRA class ID is missing from the candidate vocabulary")
            self.sample_metadata.append(metadata)
            self._labels.append(class_to_index[cid])
        self.metadata = dict(self.sample_metadata[0])

    def __len__(self) -> int:
        return len(self.paths)

    def factors(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return CPU ``(B, A, label)`` for reconstruction on the training device."""
        with safe_open(str(self.paths[index]), framework="pt", device="cpu") as handle:
            a = handle.get_tensor(self.layer_name_down)
            b = handle.get_tensor(self.layer_name_up)
        # Preserve the released float32 reconstruction protocol; silently
        # changing factor precision changes the experiment.
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("The SD-LoRA protocol requires float32 factors")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError("Non-finite LoRA factors")
        return b, a, torch.tensor(self._labels[index], dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, a, label = self.factors(index)
        weights = b @ a
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("Non-finite reconstructed LoRA matrix")
        return weights, label


class _FactorDataset(Dataset):
    """DataLoader adapter preserving factors until they reach the GPU."""

    def __init__(self, source: SDLoRADataset) -> None:
        self.source = source
        self.classes = source.classes
        self.paths = source.paths
        self.input_shape = source.input_shape
        self.layer_name = source.layer_name
        self.layer_name_down = source.layer_name_down
        self.layer_name_up = source.layer_name_up
        self.metadata = source.metadata
        self.split = source.split

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.source.factors(index)


def factor_dataset(dataset: SDLoRADataset) -> Dataset:
    """Return batches of ``(B, A, label)``; reconstruct after device transfer.

    For the paper protocol, prefer this adapter over the convenience CPU
    ``B @ A`` returned by ``SDLoRADataset.__getitem__``.
    """
    return _FactorDataset(dataset)


def load_sd_dataset(root: Path, subset: str, layer: int, split: str,
                    classes: list[str] | None = None) -> SDLoRADataset:
    """Load a local split with shared seen/held-out vocabulary by default."""
    if split not in SPLITS:
        raise ValueError(f"unknown SD-LoRA split: {split}")
    if classes is None:
        seen, heldout = discover_sd_classes(root, subset)
        classes = heldout if split.endswith("holdout") else seen
    return SDLoRADataset(root, subset, layer, split, classes)


def load_sd_reference(path: Path, layer_name: str, shape: Sequence[int]) -> torch.Tensor:
    """Load an explicitly supplied reference in the bank's B@A coordinates.

    A base UNet weight is not a reference in LoRA-update coordinates. Require
    an annotated matrix artifact so a full pretrained weight cannot silently
    be mixed with an update-only bank. The artifact contains the single tensor
    ``reference`` and metadata ``representation=lora_update`` and ``layer_name``.
    """
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("SD references must be annotated safetensors matrix artifacts")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("representation") != "lora_update":
            raise ValueError("SD reference metadata must declare representation=lora_update")
        if metadata.get("layer_name") != layer_name:
            raise ValueError("SD reference layer_name does not match the selected LoRA factor pair")
        if list(handle.keys()) != ["reference"]:
            raise ValueError("SD reference artifact must contain only the tensor 'reference'")
        reference = handle.get_tensor("reference")
    if tuple(reference.shape) != tuple(shape):
        raise ValueError("SD reference shape does not match the reconstructed B@A matrix")
    if not torch.is_floating_point(reference) or not bool(torch.isfinite(reference).all()):
        raise ValueError("SD reference must be a finite floating-point matrix")
    return reference.detach().float().contiguous()


def build_text_embeddings(classes: Sequence[str], class_names: Mapping[str, str],
                          device: str | torch.device = "cpu",
                          download_root: Path | str | None = None, *,
                          allow_download: bool = False) -> torch.Tensor:
    """Explicitly encode one candidate group using frozen CLIP ViT-L/14.

    Rows follow ``classes`` exactly. Prompts are ``a photo of a NAME``, with
    underscores converted to spaces, matching the original name mapping.
    Features are unnormalized, detached, float32 CPU tensors. To stay offline,
    place ``ViT-L-14.pt`` in ``download_root``. Network access is opt-in.
    """
    classes = list(classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Provide a nonempty, unique candidate-class sequence")
    prompts = []
    for cid in classes:
        name = class_names.get(cid)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing class name for {cid}")
        prompts.append("a photo of a " + " ".join(name.split("_")))
    cache = Path(download_root).expanduser() if download_root is not None else None
    checkpoint = cache / "ViT-L-14.pt" if cache is not None else None
    if checkpoint is not None and checkpoint.is_file():
        model_name = str(checkpoint)
    elif allow_download:
        model_name = "ViT-L/14"
    else:
        raise FileNotFoundError(
            "Provide cached ViT-L-14.pt in download_root, or explicitly set allow_download=True"
        )
    try:
        import clip
    except ImportError as error:
        raise ImportError("Install OpenAI CLIP to explicitly prepare text features") from error
    model, _ = clip.load(model_name, device=device,
                         download_root=str(cache) if cache is not None else None)
    model.eval().requires_grad_(False)
    tokens = torch.cat([clip.tokenize(prompt) for prompt in prompts], dim=0).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens).float().detach().cpu()
    if tuple(features.shape) != (len(classes), 768) or not bool(torch.isfinite(features).all()):
        raise ValueError("CLIP ViT-L/14 must return finite 768-dimensional text features")
    return features


def load_text_embeddings(seen: Sequence[str], heldout: Sequence[str],
                         device: str | torch.device = "cpu",
                         cache_dir: Path | str | None = None,
                         class_names_file: Path | str | None = None, *,
                         allow_download: bool = False) -> dict[str, torch.Tensor]:
    """Build ``{'train': seen_features, 'holdout': heldout_features}`` explicitly.

    The default JSON mapping is ``assets/imagenet_class_names.json`` within the
    package. Each group is encoded separately, as in the original protocol;
    row order follows the caller without silently sorting or normalizing it.
    """
    seen, heldout = list(seen), list(heldout)
    if not seen or not heldout or set(seen) & set(heldout):
        raise ValueError("Seen and held-out classes must be nonempty and disjoint")
    path = (Path(class_names_file) if class_names_file is not None
            else Path(__file__).resolve().parent / "assets" / "imagenet_class_names.json")
    with path.open(encoding="utf-8") as handle:
        names = json.load(handle)
    if not isinstance(names, dict):
        raise ValueError("The class-name JSON must map class IDs to names")
    return {
        group: build_text_embeddings(classes, names, device, cache_dir,
                                      allow_download=allow_download)
        for group, classes in (("train", seen), ("holdout", heldout))
    }
