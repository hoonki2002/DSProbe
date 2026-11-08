"""Seed aggregation, validation layer selection, and forward-order plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math

import numpy as np

from .experiments import COUNTS, reference_kind
from .runner import write_json

LABELS = {"probex": "ProbeX", "probex4": "ProbeX (x4)", "mvprobe": "MVProbe",
          "dsprobe": "DSProbe", "r_dsprobe": "R-DSProbe", "statnn": "StatNN",
          "statnn_fullmodel_xgboost": "StatNN (full model, XGBoost)",
          "poly6": "Degree 3", "poly8": "Degree 3,4",
          "duplicate6": "Capacity control (6)", "duplicate8": "Capacity control (8)"}
COLORS = {"probex": "#D55E00", "mvprobe": "#0072B2",
          "dsprobe": "#9467bd", "r_dsprobe": "#009E73"}
STYLES = {"probex": ":", "mvprobe": "--", "dsprobe": "-", "r_dsprobe": "-"}


def write_csv(path, rows):
    if not rows:
        Path(path).unlink(missing_ok=True)
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def forward_mapping(model):
    name = "resnet" if model == "ResNet" else "vit"
    path = Path(__file__).parent / "assets" / "layers" / f"{name}_forward.csv"
    with path.open(newline="") as stream:
        return {int(row["original_layer_index"]): int(row["plot_position"])
                for row in csv.DictReader(stream)}


def collect_results(root):
    rows, identities = [], set()
    for path in sorted(Path(root).rglob("result.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        for group, metrics in result["results"].items():
            identity = tuple(result[key] for key in ("task", "model", "layer", "method", "seed")) + (group,)
            if identity in identities:
                raise ValueError("Duplicate model/layer/method/seed/group result")
            identities.add(identity)
            row = {key: result[key] for key in ("task", "model", "layer", "method", "seed")}
            row.update(group=group, validation_accuracy=metrics["validation"]["accuracy"],
                       test_accuracy=metrics["test"]["accuracy"], best_epoch=metrics["validation"]["epoch"],
                       parameters=result["parameters"])
            if not all(math.isfinite(float(row[key])) for key in ("validation_accuracy", "test_accuracy")):
                raise ValueError("Nonfinite result")
            # Refuse aggregation of configurations that differ beyond seed.
            contract = result["contract"]
            configuration = {key: contract.get(key) for key in
                ("model", "training", "sources", "layer_name", "source_sha256", "torch_version", "device_type",
                 "initial_reference_sha256", "learned_reference_init",
                 "random_norm_target_sha256", "text_feature_sha256", "numerics")}
            if result["method"] == "statnn_fullmodel_xgboost":
                if result["layer"] != "full_model" or contract.get("input_scope") != "all_named_parameters":
                    raise ValueError("Invalid full-model StatNN result schema")
                configuration["feature_sha256"] = contract.get("feature_sha256")
            kind = reference_kind(result["method"])
            initialization = contract.get("learned_reference_init") if kind == "learned" else kind
            # Keep the original hash check for legacy norm-matched results
            # whose shared normalization target was not recorded.
            if initialization == "random" or (
                initialization == "random_norm" and contract.get("random_norm_target_sha256")
            ):
                configuration.pop("initial_reference_sha256")
            row["configuration"] = json.dumps(configuration, sort_keys=True)
            rows.append(row)
    if not rows:
        raise ValueError("No completed result.json files found")
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in ("task", "model", "layer", "method", "group"))].append(row)
    result = []
    # Full-model StatNN has no integer layer. Leave neural layer ordering intact.
    for key, items in sorted(groups.items(), key=lambda item: (
            item[0][0], item[0][1], -1 if item[0][2] == "full_model" else item[0][2],
            item[0][3], item[0][4])):
        if len({item["configuration"] for item in items}) != 1:
            raise ValueError("Cannot average seeds with different scientific configurations")
        values = np.array([[item["validation_accuracy"], item["test_accuracy"]] for item in items])
        row = dict(zip(("task", "model", "layer", "method", "group"), key))
        row.update(n_seeds=len(items), complete_five_seeds=set(item["seed"] for item in items) == {1, 2, 3, 4, 5},
                   validation_mean=float(values[:, 0].mean()), validation_sd=float(values[:, 0].std(ddof=0)),
                   test_mean=float(values[:, 1].mean()), test_sd=float(values[:, 1].std(ddof=0)),
                   parameters=items[0]["parameters"])
        result.append(row)
    return result


def _layerwise_summary(rows, output):
    best_layers, robustness = [], []
    for model, count in COUNTS.items():
        candidates = [row for row in rows if row["model"] == model and row["group"] == "id"
                      and row["layer"] != "full_model"]
        by_method = defaultdict(dict)
        for row in candidates:
            by_method[row["method"]][row["layer"]] = row
        for method, layer_rows in sorted(by_method.items()):
            expected = set(range(1 if method == "dsprobe" else 0, count))
            complete = set(layer_rows) == expected and all(row["complete_five_seeds"] for row in layer_rows.values())
            best = max(layer_rows.values(), key=lambda row: (row["validation_mean"], -row["layer"]))
            best_layers.append({**best, "full_sweep_complete": complete})
        mapping = forward_mapping(model)
        # Central 70% of the original full-network positions; never compress gaps.
        lower = math.ceil(0.15 * (count - 1))
        upper = math.floor(0.85 * (count - 1))
        intended = {layer for layer, position in mapping.items() if lower <= position <= upper}
        for method in ("dsprobe", "r_dsprobe"):
            if method not in by_method or "mvprobe" not in by_method:
                continue
            common = sorted(intended & set(by_method[method]) & set(by_method["mvprobe"]))
            if not common:
                continue
            ours = np.array([by_method[method][layer]["test_mean"] for layer in common])
            base = np.array([by_method["mvprobe"][layer]["test_mean"] for layer in common])
            complete = set(common) == intended and all(by_method[name][layer]["complete_five_seeds"]
                for name in (method, "mvprobe") for layer in common)
            robustness.append({"model": model, "method": method, "first_forward_position": lower,
                "last_forward_position": upper, "expected_layers": len(intended), "observed_layers": len(common),
                "complete": complete, "win_rate": float(100 * (ours > base).mean()),
                "method_median": float(np.median(ours)), "baseline_median": float(np.median(base)),
                "method_min": float(ours.min()), "baseline_min": float(base.min())})
    for method in ("dsprobe", "r_dsprobe"):
        architecture_rows = [row for row in robustness if row["method"] == method]
        if ({row["model"] for row in architecture_rows} != set(COUNTS)
                or not all(row["complete"] for row in architecture_rows)):
            continue
        # Macro averages weight architectures equally, not by their layer count.
        robustness.append({"model": "Average", "method": method,
            "first_forward_position": "", "last_forward_position": "",
            "expected_layers": sum(row["expected_layers"] for row in architecture_rows),
            "observed_layers": sum(row["observed_layers"] for row in architecture_rows),
            "complete": True,
            **{metric: float(np.mean([row[metric] for row in architecture_rows]))
               for metric in ("win_rate", "method_median", "baseline_median", "method_min", "baseline_min")}})
    write_csv(output / "validation_selected_layers.csv", best_layers)
    write_csv(output / "layer_robustness.csv", robustness)
    return best_layers


def _plot(rows, output, best_layers, log_forward_axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (model, count) in zip(axes.flat, COUNTS.items()):
        mapping = forward_mapping(model)
        for method in ("probex", "mvprobe", "dsprobe", "r_dsprobe"):
            relevant = [row for row in rows if row["model"] == model and row["method"] == method and row["group"] == "id"]
            if not relevant:
                continue
            y = np.full(count, np.nan)
            for row in relevant:
                y[mapping[row["layer"]]] = row["test_mean"]
            x = np.arange(count) + (1 if log_forward_axis else 0)
            ax.plot(x, y, label=LABELS[method], color=COLORS[method], linestyle=STYLES[method], linewidth=1.6)
            selected = next((row for row in best_layers if row["model"] == model and row["method"] == method and row["full_sweep_complete"]), None)
            if selected:
                position = mapping[selected["layer"]]
                ax.plot(x[position], y[position], "o", markerfacecolor="none", color=COLORS[method])
        if log_forward_axis:
            ax.set_xscale("log")
        ax.set_title(model)
        ax.set_xlabel("Forward weight position" + (" (one-based, log scale)" if log_forward_axis else " (zero-based)"))
        ax.set_ylabel("Membership accuracy (%)")
        ax.grid(alpha=0.2)
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"layerwise.{extension}", dpi=180)
    plt.close(fig)
    # Paired axis-order diagnostic uses the same rows without changing scores.
    model, count = "SupViT", COUNTS["SupViT"]
    mapping = forward_mapping(model)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for method in ("probex", "mvprobe", "dsprobe", "r_dsprobe"):
        relevant = [row for row in rows if row["model"] == model and row["method"] == method and row["group"] == "id"]
        for ax, forward in zip(axes, (False, True)):
            y = np.full(count, np.nan)
            for row in relevant:
                y[mapping[row["layer"]] if forward else row["layer"]] = row["test_mean"]
            ax.plot(np.arange(count), y, color=COLORS[method], linestyle=STYLES[method], label=LABELS[method], linewidth=1.6)
            ax.set_xlabel("Forward weight position" if forward else "Repository layer index")
            ax.set_ylabel("Membership accuracy (%)")
            ax.grid(alpha=0.2)
    fig.savefig(output / "layer_order_comparison.pdf")
    plt.close(fig)


def summarize(root, output, *, layerwise=False, log_forward_axis=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = collect_results(root)
    rows = aggregate(per_seed)
    write_csv(output / "per_seed.csv", [{key: value for key, value in row.items() if key != "configuration"} for row in per_seed])
    write_csv(output / "summary.csv", rows)
    lines = ["# Experiment results", "", "Accuracy is in percent; SD uses ddof=0.",
             "A cell is complete only when seeds 1--5 are all present.", "",
             "| Model | Layer | Method | Group | Accuracy | Seeds | Complete |",
             "|---|---:|---|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['layer']} | {LABELS.get(row['method'], row['method'])} | {row['group']} | "
                     f"{row['test_mean']:.3f} +/- {row['test_sd']:.3f} | {row['n_seeds']} | {row['complete_five_seeds']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if layerwise:
        best = _layerwise_summary(rows, output)
        _plot(rows, output, best, log_forward_axis)
    write_json(output / "coverage.json", {"seed_results": len(per_seed), "cells": len(rows),
               "complete_cells": sum(row["complete_five_seeds"] for row in rows)})
