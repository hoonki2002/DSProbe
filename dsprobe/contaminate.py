"""Tree-label noise: what if a tree's training set contains leaves of other trees?

R-DSProbe initializes W* at the mean of a tree's training leaves and trains on
them. If the tree membership is only estimated and the estimate is wrong, some
training leaves belong to another tree. This builds banks where a fraction p of
each tree's training leaves is replaced by leaves of the *other* trees, choosing
the foreign leaves nearest to the tree's training mean (the most plausible
confusions), so the harm is not understated. Validation and test stay clean.

Banks are symlink trees over an existing bank (``--data-root``), one per p, in the
usual ``MODEL/SPLIT/<leaf>`` layout, so the unchanged ``train`` command runs on
them. The replaced own leaves are drawn at random with ``--seed``; the sets are
nested across p. ``manifest.json`` records every replacement.

    python -m dsprobe.contaminate --data-root data --output data_contam --fractions 0 5 10 25 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .runner import write_json
from .tree_estimation import OWN_LAYER, SPLITS, TREES, load_leaves


def _distances(X, indices, centroid, chunk=128):
    out = []
    for start in range(0, len(indices), chunk):
        rows = torch.from_numpy(X[indices[start:start + chunk]]).double()
        out.append(torch.cdist(rows, centroid[None]).squeeze(1))
    return torch.cat(out).numpy()


def plan(data_root, fractions, seed=0, layers=OWN_LAYER, trees=TREES):
    """Which own leaves leave and which foreign leaves enter, per tree and fraction."""
    result = {}
    for a, tree in enumerate(trees):
        X, tree_ids, split_ids, names, _ = load_leaves(data_root, layers[tree], trees)
        own = np.where((tree_ids == a) & (split_ids == 0))[0]
        foreign = np.where((tree_ids != a) & (split_ids == 0))[0]
        centroid = torch.from_numpy(X[own]).double().mean(0)
        order_foreign = foreign[np.argsort(_distances(X, foreign, centroid), kind="stable")]
        distance_foreign = _distances(X, order_foreign, centroid)
        own_distance = _distances(X, own, centroid)
        order_own = np.random.default_rng(seed).permutation(own)
        entry = {"layer": layers[tree], "n_train": int(len(own)),
                 "own_train_to_mean": {"mean": float(own_distance.mean()), "max": float(own_distance.max())},
                 "fractions": {}}
        for p in fractions:
            k = int(round(p / 100 * len(own)))
            if k > len(order_foreign):
                raise ValueError(f"{tree}: only {len(order_foreign)} foreign training leaves for k={k}")
            added = order_foreign[:k]
            entry["fractions"][str(p)] = {
                "k": k,
                "removed": [names[i] for i in order_own[:k]],
                "added": [names[i] for i in added],
                "added_from": {t: int((tree_ids[added] == b).sum()) for b, t in enumerate(trees) if b != a},
                "added_distance_to_mean": ({"min": float(distance_foreign[:k].min()),
                                            "max": float(distance_foreign[:k].max())} if k else None),
            }
        result[tree] = entry
        del X
    return result


def _link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def build(data_root, output, fractions, seed=0, layers=OWN_LAYER, trees=TREES):
    data_root, output = Path(data_root), Path(output)
    planned = plan(data_root, fractions, seed, layers, trees)
    for p in fractions:
        root = output / f"p{p}"
        for tree in trees:
            entry = planned[tree]["fractions"][str(p)]
            removed = {Path(n).name for n in entry["removed"]}
            for split in SPLITS:
                for leaf in sorted((data_root / tree / split).iterdir()):
                    if split == "train" and leaf.name in removed:
                        continue
                    _link(leaf, root / tree / split / leaf.name)
            for name in entry["added"]:
                source_tree, split, leaf = name.split("/")
                _link(data_root / source_tree / split / leaf, root / tree / "train" / leaf)
    write_json(output / "manifest.json", {"seed": seed, "fractions": list(fractions), "trees": planned})
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data_contam")
    parser.add_argument("--fractions", type=int, nargs="+", default=[0, 5, 10, 25, 50],
                        help="percent of each tree's training leaves replaced")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    planned = build(args.data_root, args.output, args.fractions, args.seed)
    for tree, entry in planned.items():
        for p, f in entry["fractions"].items():
            print(f"{tree:>6} p={p:>2}%  replaced {f['k']:>3} of {entry['n_train']}  from {f['added_from']}  "
                  f"distance to own mean {f['added_distance_to_mean']}", flush=True)
    print(f"-> {Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()
