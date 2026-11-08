"""Does R-DSProbe need the tree labels? Estimate trees from weights alone.

R-DSProbe keeps one reference W* per tree, initialized at the mean of that
tree's training leaves, so it needs to know which models belong to which tree.
Model Jungle supplies that grouping; a model collected elsewhere would not. This
module asks whether the grouping can be recovered from the weights.

``distance``  Frobenius distance from every leaf to the pretrained root of every
              tree (no training), absolute and relative. Leaves of tree A against
              root A is the within-tree distance; against roots B and C it is the
              cross-tree distance. The separation is judged on absolute distance
              (see run_distance). Roots must be the pretrained backbones in
              --reference-root.
``cluster``   Mix the leaves of all trees without their labels, run k-means on one
              flattened layer, and compare the clusters with the true trees:
              purity, adjusted Rand index, and confusion matrix. Fitted once on
              every split (the requested protocol) and once on train only with
              val/test assigned to the nearest centroid (an unseen model).

All three trees are ViT-B, so a common layer index has one shape and the leaves
can share one clustering space. Layer ids are Model-J repository indices.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .modelj_data import ModelJDataset, load_reference, matrix_from_tensor, resolve_layer_name
from .runner import _reference_path, write_json

TREES = ("SupViT", "MAE", "DINO")
OWN_LAYER = {"SupViT": 53, "MAE": 65, "DINO": 47}
SPLITS = ("train", "val", "test")


def relative_distance(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    matrix = matrix.detach().to(torch.float64)
    reference = reference.detach().to(torch.float64)
    return float(torch.linalg.matrix_norm(matrix - reference)
                 / torch.linalg.matrix_norm(reference) * 100)


def purity(truth, assigned) -> float:
    """Fraction of leaves that carry the majority true tree of their cluster."""
    truth, assigned = np.asarray(truth), np.asarray(assigned)
    return float(sum(np.bincount(truth[assigned == c]).max()
                     for c in np.unique(assigned)) / len(truth))


def confusion(truth, assigned, n_trees, n_clusters) -> list[list[int]]:
    table = np.zeros((n_trees, n_clusters), dtype=int)
    for t, c in zip(truth, assigned):
        table[t, c] += 1
    return table.tolist()


def load_leaves(data_root, layer: int, trees=TREES):
    """Flattened layer ``layer`` of every leaf of ``trees``, labels kept aside."""
    rows, tree_ids, split_ids, names = [], [], [], []
    shape = None
    for t, tree in enumerate(trees):
        key = resolve_layer_name(tree, layer)
        for s, split in enumerate(SPLITS):
            for path in ModelJDataset(data_root, tree, layer, split).paths:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    matrix = matrix_from_tensor(handle.get_tensor(key), tree)
                if shape is None:
                    shape = tuple(matrix.shape)
                elif tuple(matrix.shape) != shape:
                    raise ValueError(f"{tree} L{layer} has shape {tuple(matrix.shape)}, expected {shape}")
                rows.append(matrix.reshape(-1))
                tree_ids.append(t)
                split_ids.append(s)
                names.append(f"{tree}/{split}/{path.parents[2].name}")
    return (torch.stack(rows).numpy(), np.asarray(tree_ids), np.asarray(split_ids), names, shape)


def _stats(values):
    return {"n": len(values), "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def run_distance(data_root, reference_root, layers, output, trees=TREES):
    """Leaf-to-root distances, absolute and relative, with the worst-case separation.

    The relative distance divides by the norm of the root it is measured to, and
    the three roots differ a lot in norm, so within-tree and cross-tree relative
    distances are not comparable. The separation is therefore judged on the
    absolute Frobenius distance: ``gap`` is the smallest distance of any leaf to
    a foreign root over the largest distance of any leaf to its own root.
    """
    output = Path(output)
    rows, summary = [], {}
    for layer in layers:
        X, tree_ids, split_ids, _, shape = load_leaves(data_root, layer, trees)
        key = resolve_layer_name(trees[0], layer)
        roots = [load_reference(_reference_path(reference_root, tree, layer), tree, key, shape)
                 for tree in trees]
        norms = np.array([float(torch.linalg.matrix_norm(r.to(torch.float64))) for r in roots])
        absolute = np.array([[float(torch.linalg.matrix_norm(
            torch.from_numpy(x).reshape(shape).to(torch.float64) - root.to(torch.float64)))
            for root in roots] for x in X])
        relative = absolute / norms * 100
        nearest = absolute.argmin(1)
        entry = {"layer": layer, "layer_name": key, "root_norms": norms.tolist(),
                 "pairs": {}, "per_tree": {}}
        for a, leaf_tree in enumerate(trees):
            for b, root_tree in enumerate(trees):
                mask = tree_ids == a
                rows.append({"layer": layer, "leaf_tree": leaf_tree, "root_tree": root_tree,
                             "kind": "within" if a == b else "cross",
                             **{f"abs_{k}": v for k, v in _stats(absolute[mask, b].tolist()).items()},
                             **{f"rel_pct_{k}": v for k, v in _stats(relative[mask, b].tolist()).items()}})
                entry["pairs"][f"{leaf_tree}->{root_tree}"] = {
                    "absolute": _stats(absolute[mask, b].tolist()),
                    "relative_pct": _stats(relative[mask, b].tolist())}
            mask = tree_ids == a
            own_max = float(absolute[mask, a].max())
            foreign_min = float(np.delete(absolute[mask], a, axis=1).min())
            entry["per_tree"][leaf_tree] = {"max_own_root": own_max, "min_foreign_root": foreign_min,
                                            "gap": foreign_min / own_max}
        entry["nearest_root_accuracy"] = float((nearest == tree_ids).mean())
        entry["worst_gap"] = min(v["gap"] for v in entry["per_tree"].values())
        entry["worst_leaf_margin_abs"] = float(min(
            np.delete(absolute[i], tree_ids[i]).min() - absolute[i, tree_ids[i]] for i in range(len(X))))
        summary[str(layer)] = entry
        print(f"L{layer} {key}: nearest-root accuracy {entry['nearest_root_accuracy']:.4f}  "
              f"worst gap x{entry['worst_gap']:.2f}  worst leaf margin {entry['worst_leaf_margin_abs']:+.2f}  "
              f"root norms {np.round(norms, 2).tolist()}", flush=True)
        for name, v in entry["per_tree"].items():
            print(f"    {name:>6}: farthest leaf from own root {v['max_own_root']:6.2f} | "
                  f"nearest leaf to a foreign root {v['min_foreign_root']:6.2f} (x{v['gap']:.2f})", flush=True)
        del X, absolute, relative
    output.mkdir(parents=True, exist_ok=True)
    with (output / "distance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "distance.json", summary)
    print(f"-> {output / 'distance.csv'}")
    return summary


def _kmeans():
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
    except ImportError as error:
        raise RuntimeError("clustering needs scikit-learn: pip install '.[statnn]'") from error
    return KMeans, adjusted_rand_score


def run_cluster(data_root, layers, output, seeds=(0, 1, 2, 3, 4), n_init=10, trees=TREES):
    KMeans, adjusted_rand_score = _kmeans()
    output = Path(output)
    k = len(trees)
    summary = {}
    for layer in layers:
        X, tree_ids, split_ids, names, shape = load_leaves(data_root, layer, trees)
        train = split_ids == 0
        layer_result = {"layer": layer, "layer_name": resolve_layer_name(trees[0], layer),
                        "n_leaves": int(len(X)), "per_seed": []}
        assignments = None
        for seed in seeds:
            full = KMeans(k, n_init=n_init, random_state=seed).fit(X)
            fitted = full.labels_
            held = KMeans(k, n_init=n_init, random_state=seed).fit(X[train])
            predicted = held.predict(X[~train])
            record = {
                "seed": seed,
                "all_purity": purity(tree_ids, fitted),
                "all_ari": float(adjusted_rand_score(tree_ids, fitted)),
                "all_confusion": confusion(tree_ids, fitted, k, k),
                "train_fit_train_purity": purity(tree_ids[train], held.labels_),
                "heldout_purity": purity(tree_ids[~train], predicted),
                "heldout_ari": float(adjusted_rand_score(tree_ids[~train], predicted)),
                "heldout_confusion": confusion(tree_ids[~train], predicted, k, k),
            }
            layer_result["per_seed"].append(record)
            if assignments is None:
                assignments = [{"leaf": n, "tree": trees[t], "split": SPLITS[s], "cluster": int(c)}
                               for n, t, s, c in zip(names, tree_ids, split_ids, fitted)]
            print(f"L{layer} seed {seed}: purity(all)={record['all_purity']:.4f} "
                  f"ARI={record['all_ari']:.4f}  held-out purity={record['heldout_purity']:.4f}",
                  flush=True)
        for field in ("all_purity", "heldout_purity"):
            values = [r[field] for r in layer_result["per_seed"]]
            layer_result[f"{field}_min"], layer_result[f"{field}_mean"] = min(values), statistics.fmean(values)
        summary[str(layer)] = layer_result
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / f"cluster_assignments_L{layer}.json", assignments)
        del X
    write_json(output / "cluster.json", summary)
    with (output / "cluster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "seed", "all_purity", "all_ari", "heldout_purity", "heldout_ari"])
        for layer_result in summary.values():
            for r in layer_result["per_seed"]:
                writer.writerow([layer_result["layer"], r["seed"], r["all_purity"], r["all_ari"],
                                 r["heldout_purity"], r["heldout_ari"]])
    print(f"-> {output / 'cluster.csv'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("distance", "cluster", "all"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--reference-root", default="references")
    parser.add_argument("--output", default="results/tree_estimation")
    parser.add_argument("--layers", type=int, nargs="+", default=[53, 65, 47],
                        help="Model-J layer ids; the default is each tree's own probing layer")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="k-means seeds")
    parser.add_argument("--n-init", type=int, default=10)
    args = parser.parse_args(argv)
    if args.step in ("distance", "all"):
        run_distance(args.data_root, args.reference_root, args.layers, args.output)
    if args.step in ("cluster", "all"):
        run_cluster(args.data_root, args.layers, args.output, tuple(args.seeds), args.n_init)


if __name__ == "__main__":
    main()
