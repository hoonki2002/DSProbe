"""Opt-in acquisition of the official Model Jungle splits and frozen roots.

Run ``python -m dsprobe.download --help``. Importing this module, requesting
help, and planning with ``--dry-run`` never import download dependencies or
contact a service. Checkpoint tensors are not executed by this downloader.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import sys


MODELJ = ("ResNet", "SupViT", "MAE", "DINO")
SD = ("SD_200", "SD_1k")
SPLITS = ("train", "val", "test")
SD_SPLITS = SPLITS + ("val_holdout", "test_holdout")
INDEX_REPO = "ProbeX/Model-J"
REFERENCES = {
    "ResNet": ("microsoft/resnet-101", "c2bf50f68263a35f102eb5c84ca91fc7352ceff3",
               "pytorch_model.bin", "ResNet.bin"),
    "SupViT": ("google/vit-base-patch16-224", "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
               "model.safetensors", "SupViT.safetensors"),
    "MAE": ("facebook/vit-mae-base", "25b184bea5538bf5c4c852c79d221195fdd2778d",
            "model.safetensors", "MAE.safetensors"),
    "DINO": ("facebook/dino-vitb16", "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2",
             "pytorch_model.bin", "DINO.bin"),
}


@dataclass(frozen=True)
class Download:
    subset: str
    split: str
    repo_id: str
    filename: str
    revision: str
    destination: str


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.startswith("~"):
        raise argparse.ArgumentTypeError("Use a relative path, for example data or references")
    return path


def _safe_repo(repo_id: str) -> str:
    parts = PurePosixPath(repo_id).parts
    if len(parts) != 2 or any(part in (".", "..") for part in parts) or "\\" in repo_id:
        raise ValueError("Invalid checkpoint repository ID in the official index")
    return repo_id


def _sd_path(row, split: str) -> str:
    name = row.get("hf_model_path")
    if not name:
        index = int(row["model_idx"])
        if index < 0:
            raise ValueError("Negative model index")
        name = f"{split}/model_idx_{index:04d}/pytorch_lora_weights.safetensors"
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or
            len(path.parts) < 3 or path.parts[0] != split or
            path.name != "pytorch_lora_weights.safetensors"):
        raise ValueError("Invalid SD checkpoint path or split in the official index")
    return str(path)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _relative_link(source: Path, destination: Path, force: bool) -> None:
    """Expose a cached file without a second checkpoint copy or absolute link."""
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source:
            return
        if not force:
            raise FileExistsError(f"Destination already exists: {destination}; use --force to replace it")
        if destination.is_dir():
            raise IsADirectoryError(str(destination))
    temporary = destination.with_name(destination.name + ".partial-link")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(source, destination.parent.resolve()))
    temporary.replace(destination)


def _download_one(job: Download, cache: Path, force: bool, hf_hub_download) -> dict:
    source = Path(hf_hub_download(
        repo_id=job.repo_id, repo_type="model", filename=job.filename,
        revision=job.revision, cache_dir=str(cache / "hub"), force_download=force,
    ))
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Downloaded checkpoint is empty or absent")
    revision = source.parent.name if job.revision == "main" else job.revision
    destination = Path(job.destination)
    if job.subset in MODELJ and job.split != "reference":
        # hf_hub_download returns .../snapshots/COMMIT/model.safetensors.
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError("Could not recover the downloaded checkpoint revision")
        from safetensors import safe_open
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            if "dataset_chosen_targets" not in (handle.metadata() or {}):
                raise ValueError("Checkpoint lacks membership-label metadata")
        snapshot = destination / "snapshots" / revision / "model.safetensors"
        _relative_link(source, snapshot, force)
        pointer = destination / "refs" / "main"
        if pointer.is_file() and pointer.read_text(encoding="utf-8").strip() != revision and not force:
            raise FileExistsError("Checkpoint revision changed; use --force in a separate data bank")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_name("main.partial")
        temporary.write_text(revision + "\n", encoding="utf-8")
        temporary.replace(pointer)
    else:
        if job.subset in SD:
            from safetensors import safe_open
            with safe_open(str(source), framework="pt", device="cpu") as handle:
                if not (handle.metadata() or {}).get("imagenet_class_id"):
                    raise ValueError("LoRA checkpoint lacks class metadata")
        _relative_link(source, destination, force)
    return {**asdict(job), "resolved_revision": revision, "bytes": source.stat().st_size}


def _plan_subset(subset, data_root, cache, revision, load_dataset, api):
    index = load_dataset(INDEX_REPO, subset, revision=revision, cache_dir=str(cache / "datasets"))
    required = SD_SPLITS if subset in SD else SPLITS
    missing = set(required) - set(index)
    if missing:
        raise ValueError(f"Official index lacks required splits: {sorted(missing)}")
    jobs, seen = [], set()
    sd_revision = api.model_info(f"ProbeX/Model-J__{subset}").sha if subset in SD else None
    for split in required:
        if not len(index[split]):
            raise ValueError(f"Official split is empty: {subset}/{split}")
        for row in index[split]:
            if subset in SD:
                filename = _sd_path(row, split)
                repo_id = f"ProbeX/Model-J__{subset}"
                destination = data_root / subset / filename
                item_revision = sd_revision
            else:
                repo_id = _safe_repo(row["hf_model_id"])
                filename = "model.safetensors"
                destination = data_root / subset / split / ("models--" + repo_id.replace("/", "--"))
                item_revision = "main"
            identity = (repo_id, filename)
            if identity in seen:
                raise ValueError("Official index repeats a checkpoint; refusing duplicate or overlapping splits")
            seen.add(identity)
            jobs.append(Download(subset, split, repo_id, filename, item_revision, str(destination)))
    return jobs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--subset", nargs="+", choices=MODELJ + SD,
                        help="Official checkpoint banks to fetch; all released splits are preserved")
    result.add_argument("--references", action="store_true", help="Fetch frozen pretrained roots")
    result.add_argument("--reference-models", nargs="+", choices=MODELJ,
                        help="Roots to fetch; defaults to selected Model-J subsets, or all four")
    result.add_argument("--data-root", type=relative_path, default=Path("data"))
    result.add_argument("--reference-root", type=relative_path, default=Path("references"))
    result.add_argument("--index-revision", default="main", help="Model-J index revision to resolve and record")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--force", action="store_true", help="Redownload and replace conflicting local destinations")
    result.add_argument("--allow-download", action="store_true", help="Explicitly allow all network acquisition")
    result.add_argument("--dry-run", action="store_true", help="Print sources and destinations without network or writes")
    return result


def main(argv=None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if not args.subset and not args.references:
        arguments.error("Choose --subset and/or --references")
    if args.reference_models and not args.references:
        arguments.error("--reference-models requires --references")
    if args.workers < 1:
        arguments.error("--workers must be positive")
    subsets = list(dict.fromkeys(args.subset or []))
    roots = list(dict.fromkeys(args.reference_models or [s for s in subsets if s in MODELJ] or MODELJ))
    cache = args.data_root / ".cache"
    plan = {
        "index": INDEX_REPO, "index_revision": args.index_revision,
        "banks": [{"subset": subset, "destination": str(args.data_root / subset),
                   "splits": SD_SPLITS if subset in SD else SPLITS,
                   "repository": f"ProbeX/Model-J__{subset}" if subset in SD else "individual hf_model_id entries"}
                  for subset in subsets],
        "references": [{"model": model, "repo_id": REFERENCES[model][0],
                        "revision": REFERENCES[model][1], "filename": REFERENCES[model][2],
                        "destination": str(args.reference_root / REFERENCES[model][3])}
                       for model in roots] if args.references else [],
        "cache": str(cache),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.allow_download:
        arguments.error("Network acquisition is opt-in: add --allow-download, or inspect with --dry-run")
    print("WARNING: full checkpoint banks can require more than a terabyte. No training images are downloaded.", flush=True)
    # Set cache locations before importing Hugging Face libraries. Authentication
    # may be supplied by the standard HF_TOKEN environment variable.
    for name, path in {"HF_HOME": cache, "HF_HUB_CACHE": cache / "hub",
                       "HF_DATASETS_CACHE": cache / "datasets", "HF_MODULES_CACHE": cache / "modules"}.items():
        os.environ[name] = str(path)
    try:
        from huggingface_hub import HfApi, hf_hub_download
        if subsets:
            from datasets import load_dataset
    except ImportError as error:
        print(f"Missing optional download dependencies: {error}. Install datasets and huggingface_hub.", file=sys.stderr)
        return 2
    report = {"plan": plan, "completed": [], "failures": []}
    jobs = []
    if subsets:
        try:
            api = HfApi()
            revision = api.dataset_info(INDEX_REPO, revision=args.index_revision).sha
            report["index_resolved_revision"] = revision
            for subset in subsets:
                jobs.extend(_plan_subset(subset, args.data_root, cache, revision, load_dataset, api))
        except Exception as error:
            report["failures"].append({"stage": "official_index", "error": str(error)})
    if args.references:
        for model in roots:
            repo_id, revision, filename, local_name = REFERENCES[model]
            jobs.append(Download(model, "reference", repo_id, filename, revision,
                                 str(args.reference_root / local_name)))
    print(f"Planned {len(jobs)} checkpoint files; failures are reported, never silently omitted.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, job, cache, args.force, hf_hub_download): job for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                report["completed"].append(future.result())
            except Exception as error:
                report["failures"].append({**asdict(job), "error": str(error)})
                print(f"FAILED {job.subset}/{job.split} {job.repo_id}/{job.filename}: {error}", file=sys.stderr)
            if number % 25 == 0 or number == len(jobs):
                print(f"Processed {number}/{len(jobs)}; failures={len(report['failures'])}", flush=True)
    report["complete"] = not report["failures"]
    # Runtime acquisition records live with the data, never in package sources.
    record = args.data_root / "download_manifest.json"
    _atomic_json(record, report)
    print(f"Manifest: {record}; complete={report['complete']}", flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
