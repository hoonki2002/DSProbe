"""Command-line entry points. All user data and output paths are explicit."""

from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

import torch
from safetensors.torch import save_file

from .experiments import configurations, jobs_from_suite, read_suite
from .runner import load_modelj_inputs, load_sd_inputs, run_job, write_json
from .sd_data import discover_sd_classes, load_text_embeddings


def _csv(text):
    return text.split(",")


def _ints(text):
    return [int(value) for value in text.split(",")]


def _model_layers(items):
    if items is None:
        return None
    mapping = {}
    for item in items:
        try:
            model, values = item.split("=", 1)
            layers = _ints(values)
        except (ValueError, TypeError) as error:
            raise ValueError("Use --model-layers ResNet=98 SupViT=53 MAE=65 DINO=47") from error
        if not model or not layers or model in mapping:
            raise ValueError("Each --model-layers architecture must appear exactly once")
        mapping[model] = layers
    return mapping


def parser():
    result = argparse.ArgumentParser(description="Displacement probing experiments")
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a configured experiment suite")
    train.add_argument("--suite", type=Path, required=True)
    train.add_argument("--data-root", type=Path, default=Path("data"))
    train.add_argument("--reference-root", type=Path)
    train.add_argument("--text-root", type=Path)
    train.add_argument("--output", type=Path, default=Path("results"))
    train.add_argument("--device", help="Default: cuda:0 for neural probes; cpu for full-model StatNN")
    train.add_argument("--models", type=_csv)
    train.add_argument("--methods", type=_csv)
    train.add_argument("--layers", type=_ints)
    train.add_argument("--model-layers", nargs="+", metavar="MODEL=LAYER[,LAYER]",
                       help="Override layers separately per architecture, for the selected methods")
    train.add_argument("--seeds", type=_ints)
    train.add_argument("--epochs", type=int, help="Explicitly override epochs; recorded in every result")
    train.add_argument("--n-probes", "--k", dest="n_probes", type=int,
                       help="Base K (default 128); probex4 uses 4*K, all other branches use K")
    train.add_argument("--proj-dim", type=int, help="Features per probe (default 128)")
    train.add_argument("--rep-dim", type=int, help="Fused representation dimension (default 512)")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--eval-every", type=int, help="Validation interval in epochs")
    train.add_argument("--cap", type=float, help="Reference relative step cap (default 1e-3); 0 disables it")
    train.add_argument("--learned-init",
                       choices=["mean", "pretrained", "random", "random_norm"],
                       help="Where R-DSProbe's trainable reference starts (default: the training-bank mean). "
                            "'pretrained' and 'random_norm' read the pretrained reference, so they need --reference-root")
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--save-weights", action="store_true", help="Opt in to saving validation-selected model weights")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--dry-run", action="store_true", help="Print the experiment grid without reading data or training")
    prepare = commands.add_parser("prepare-text", help="Prepare frozen CLIP candidate embeddings")
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--subset", choices=("SD_200", "SD_1k"), required=True)
    prepare.add_argument("--output", type=Path, default=Path("references/text"))
    prepare.add_argument("--clip-cache", type=Path, required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--allow-download", action="store_true")
    analyze = commands.add_parser("analyze", help="Aggregate results and make layer-wise figures")
    analyze.add_argument("--results", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--layerwise", action="store_true")
    analyze.add_argument("--log-forward-axis", action="store_true")
    costs = commands.add_parser("costs", help="Parameter and reference-inclusive matrix-multiplication cost analysis")
    costs.add_argument("--output", type=Path, required=True)
    costs.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for reference-inclusive FLOP accounting (default: 128)")
    distance = commands.add_parser(
        "distance", help="Distance of the learned reference from the pretrained weight")
    distance.add_argument("--results", type=Path, required=True,
                          help="Results written with --save-weights")
    distance.add_argument("--data-root", type=Path, default=Path("data"))
    distance.add_argument("--reference-root", type=Path, default=Path("references"))
    distance.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        raw_suite = json.loads(args.suite.read_text(encoding="utf-8"))
        if raw_suite.get("input_scope") == "all_named_parameters":
            from .statnn import run_suite
            return run_suite(args, raw_suite)
        args.device = args.device or "cuda:0"
        args.cap = 1e-3 if args.cap is None else args.cap
        args.learned_init = args.learned_init or "mean"
        suite = read_suite(args.suite)
        jobs = list(jobs_from_suite(suite, models=args.models, methods=args.methods,
                                   layers=args.layers, seeds=args.seeds,
                                   model_layers=_model_layers(args.model_layers)))
        overrides = {key: getattr(args, key) for key in (
            "n_probes", "proj_dim", "rep_dim", "batch_size", "learning_rate",
            "weight_decay", "eval_every", "epochs")}
        if args.dry_run:
            for job in jobs:
                model, training = configurations(job, (2, 3), cap=args.cap,
                    **overrides)
                print(f"{job.key}: K={model.n_probes}, projection={model.proj_dim}, "
                      f"representation={model.rep_dim}, epochs={training.epochs}, "
                      f"lr={training.learning_rate}, weight_decay={training.weight_decay}")
            print(f"{len(jobs)} runs")
            return
        if args.num_workers < 0 or args.threads < 1:
            raise ValueError("Invalid worker/thread count")
        torch.set_num_threads(args.threads)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a compatible PyTorch wheel or pass --device cpu")
        # Share only input banks, never model/reference parameters, across jobs.
        jobs.sort(key=lambda job: (job.task, job.model, job.layer, job.method, job.seed))
        for _, group in groupby(jobs, key=lambda job: (job.task, job.model, job.layer)):
            group = list(group)
            first = group[0]
            prepared = (load_modelj_inputs(first, args.data_root) if first.task == "membership"
                        else load_sd_inputs(first, args.data_root, args.text_root))
            for job in group:
                print(f"running: {job.key}", flush=True)
                run_job(job, data_root=args.data_root, reference_root=args.reference_root,
                        text_root=args.text_root, output=args.output, device=args.device,
                        cap=args.cap,
                        learned_init=args.learned_init,
                        save_weights=args.save_weights, resume=args.resume,
                        num_workers=args.num_workers, prepared=prepared, **overrides)
            del prepared
    elif args.command == "prepare-text":
        seen, heldout = discover_sd_classes(args.data_root, args.subset)
        folder = args.output / args.subset
        if (folder / "text_embeddings.safetensors").exists() or (folder / "classes.json").exists():
            raise FileExistsError("Text features already exist; choose a new output directory")
        embeddings = load_text_embeddings(seen, heldout, args.device, args.clip_cache,
                                           allow_download=args.allow_download)
        folder.mkdir(parents=True, exist_ok=True)
        save_file(embeddings, str(folder / "text_embeddings.safetensors"))
        write_json(folder / "classes.json", {"train": seen, "holdout": heldout,
                    "clip_model": "ViT-L/14", "prompt": "a photo of a {class_name}"})
    elif args.command == "analyze":
        from .analysis import summarize
        summarize(args.results, args.output, layerwise=args.layerwise,
                  log_forward_axis=args.log_forward_axis)
    elif args.command == "distance":
        from .distance import report_distances
        report_distances(args.results, args.data_root, args.reference_root, args.output)
    else:
        from .costs import report_costs
        report_costs(args.output, batch_size=args.batch_size)
