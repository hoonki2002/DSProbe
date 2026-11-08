"""Fetch only selected layers of the Model-J checkpoints (a slim bank).

A full ViT bank is about a terabyte because every leaf is a whole backbone. The
tree-estimation experiment and the fixed-layer probing runs read one or a few
layers per leaf, so this reads each checkpoint's safetensors header and
downloads just those tensors with HTTP range requests, then writes a small
``model.safetensors`` in the same ``MODEL/SPLIT/models--OWNER--NAME/refs/main`` +
``snapshots/REV/model.safetensors`` layout as the official downloader, with the
original metadata (``dataset_chosen_targets``) preserved. The bank splits come
from the same official index. Everything reads it through ``--data-root`` as
usual, but only for the layers fetched here.

``python -m dsprobe.download_layers --subset SupViT MAE DINO --layers 53 65 47 --dry-run``
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import struct
import sys
import time
import urllib.request

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .download import INDEX_REPO, MODELJ, SPLITS, _atomic_json, _plan_subset, relative_path
from .modelj_data import resolve_layer_name

# Union over the methods of the main results (ProbeX, MVProbe, DSProbe, R-DSProbe)
# of the layer each one probes, per architecture.
MAIN_LAYERS = {"ResNet": [55, 98], "SupViT": [53, 59], "MAE": [64, 65], "DINO": [47, 59]}
DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
FIRST_READ = 1 << 16


def read_layers(fetch, names):
    """Return ``(metadata, {name: tensor})`` using ``fetch(start, end_inclusive) -> bytes``."""
    head = fetch(0, FIRST_READ - 1)
    (size,) = struct.unpack("<Q", head[:8])
    if size > 100 * 1024 * 1024:
        raise ValueError("Implausible safetensors header size")
    raw = head[8:8 + size] if size + 8 <= len(head) else fetch(8, 8 + size - 1)
    header = json.loads(raw.decode("utf-8"))
    metadata = dict(header.get("__metadata__") or {})
    base = 8 + size
    tensors = {}
    for name in names:
        if name not in header:
            raise KeyError(f"checkpoint lacks {name}")
        entry = header[name]
        if entry["dtype"] not in DTYPES:
            raise ValueError(f"unsupported dtype {entry['dtype']} for {name}")
        begin, end = entry["data_offsets"]
        data = bytearray(fetch(base + begin, base + end - 1))
        if len(data) != end - begin:
            raise ValueError(f"short read for {name}: {len(data)} of {end - begin} bytes")
        tensors[name] = torch.frombuffer(data, dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"]).clone()
    return metadata, tensors


def _retry_wait(error, attempt):
    """Seconds to wait before retrying, or None when the error is not worth retrying.

    The Hub allows a limited number of requests per window and answers 429 with
    ``Retry-After``; honour it instead of failing the checkpoint.
    """
    text = repr(error)
    header = getattr(getattr(error, "headers", None), "get", lambda *_: None)("Retry-After")
    found = re.search(r"[Rr]etry after (\d+) seconds", text)
    if "429" in text or getattr(error, "code", None) == 429:
        return min(300, int(header or (found.group(1) if found else 0) or 30) + 2)
    if attempt < 3:
        return 2 ** attempt
    return None


def _call(function, *args, attempts=12, **kwargs):
    for attempt in range(attempts):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            wait = _retry_wait(error, attempt)
            if wait is None or attempt == attempts - 1:
                raise
            time.sleep(wait)


def _http_fetch(url):
    def fetch(start, end):
        def once():
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        return _call(once)
    return fetch


def _commit(repo_id, filename, get_hf_file_metadata, hf_hub_url):
    info = _call(get_hf_file_metadata, hf_hub_url(repo_id, filename), timeout=60)
    return info.commit_hash


def _complete(snapshot: Path, names):
    if not snapshot.is_file():
        return False
    try:
        with safe_open(str(snapshot), framework="pt", device="cpu") as handle:
            return set(names) <= set(handle.keys()) and "dataset_chosen_targets" in (handle.metadata() or {})
    except Exception:
        return False


def _one(job, names, helpers):
    get_hf_file_metadata, hf_hub_url = helpers
    destination = Path(job.destination)
    pointer = destination / "refs" / "main"
    if pointer.is_file():
        revision = pointer.read_text(encoding="utf-8").strip()
        if _complete(destination / "snapshots" / revision / "model.safetensors", names):
            return {"repo": job.repo_id, "split": job.split, "status": "present"}
    revision = _commit(job.repo_id, job.filename, get_hf_file_metadata, hf_hub_url)
    if len(revision) != 40:
        raise ValueError("Could not resolve the checkpoint revision")
    url = hf_hub_url(job.repo_id, job.filename, revision=revision)
    metadata, tensors = read_layers(_http_fetch(url), names)
    if "dataset_chosen_targets" not in metadata:
        raise ValueError("Checkpoint lacks membership-label metadata")
    snapshot = destination / "snapshots" / revision / "model.safetensors"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    partial = snapshot.with_name(snapshot.name + ".partial")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(partial), metadata=metadata)
    partial.replace(snapshot)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name("main.partial")
    temporary.write_text(revision + "\n", encoding="utf-8")
    temporary.replace(pointer)
    return {"repo": job.repo_id, "split": job.split, "status": "fetched", "revision": revision,
            "bytes": sum(v.numel() * v.element_size() for v in tensors.values())}


def plan_names(subsets, layers, model_layers=None):
    """Tensor names to keep per subset: ``layers`` for all, overridden by ``MODEL=L[,L]`` items."""
    chosen = {subset: list(layers) for subset in subsets}
    for item in model_layers or []:
        model, separator, value = item.partition("=")
        if not separator or model not in chosen:
            raise ValueError(f"--model-layers entries look like MODEL=LAYER[,LAYER] for a selected subset: {item!r}")
        chosen[model] = [int(v) for v in value.split(",")]
    return {subset: sorted({resolve_layer_name(subset, layer) for layer in values})
            for subset, values in chosen.items()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=["main"],
                        help="main: every layer probed by the methods of the main results (all four architectures "
                             "unless --subset is given); replaces --layers and --model-layers")
    parser.add_argument("--subset", nargs="+", choices=MODELJ,
                        help="Default: SupViT MAE DINO, or all four with --preset main")
    parser.add_argument("--layers", nargs="+", type=int, default=[53, 65, 47],
                        help="Model-J layer ids to keep (default: each ViT tree's probing layer)")
    parser.add_argument("--model-layers", nargs="+", metavar="MODEL=LAYER[,LAYER]",
                        help="Per-subset layers replacing --layers, e.g. ResNet=55 SupViT=59 MAE=64 DINO=47")
    parser.add_argument("--data-root", type=relative_path, default=Path("data"))
    parser.add_argument("--index-revision", default="main")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent checkpoints; each costs about three Hub requests (anonymous limit: 3000 per 5 minutes)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N leaves per split (smoke test)")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.preset:
        if args.model_layers:
            raise SystemExit("--preset replaces --model-layers; use one of them")
        args.subset = args.subset or list(MAIN_LAYERS)
        args.model_layers = [f"{m}={','.join(map(str, MAIN_LAYERS[m]))}" for m in args.subset]
        args.layers = []
    args.subset = args.subset or ["SupViT", "MAE", "DINO"]
    names = plan_names(args.subset, args.layers, args.model_layers)
    if args.dry_run or not args.allow_download:
        print(json.dumps({"subsets": args.subset, "layers": args.layers, "tensors": names,
                          "data_root": str(args.data_root), "index": INDEX_REPO,
                          "note": "pass --allow-download to fetch"}, indent=2))
        return
    from datasets import load_dataset
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    cache = args.data_root / ".cache"
    jobs = []
    for subset in args.subset:
        planned = _plan_subset(subset, args.data_root, cache, args.index_revision, load_dataset, None)
        if args.limit is not None:
            planned = [j for split in SPLITS for j in [x for x in planned if x.split == split][:args.limit]]
        jobs += planned
    print(f"{len(jobs)} checkpoints; tensors: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in names.items()),
          flush=True)
    results, failures, started = [], [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_one, job, names[job.subset], (get_hf_file_metadata, hf_hub_url)): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"repo": job.repo_id, "split": job.split, "error": repr(error)})
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  failures={len(failures)}  {time.time() - started:.0f}s", flush=True)
    _atomic_json(args.data_root / "download_layers_manifest.json",
                 {"layers": args.layers, "model_layers": args.model_layers, "tensors": names, "fetched": results, "failures": failures})
    print(f"Done: fetched/present={len(results)} failures={len(failures)}")
    if failures:
        print(json.dumps(failures[:5], indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
