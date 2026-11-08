"""Portable CPU checks for the label-free tree-estimation experiment."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from dsprobe import tree_estimation as te
from dsprobe.modelj_data import CIFAR100_CLASSES, resolve_layer_name


def build(root: Path, layer: int, noise: float, generator, sizes=(("train", 6), ("val", 3), ("test", 3))):
    """Three trees whose leaves are small perturbations of three distinct roots."""
    roots = {}
    for tree in te.TREES:
        key = resolve_layer_name(tree, layer)
        roots[tree] = torch.randn(5, 7, generator=generator)
        (root / "references").mkdir(exist_ok=True)
        save_file({key: roots[tree]}, str(root / "references" / f"{tree}.safetensors"))
        for split, count in sizes:
            for index in range(count):
                checkpoint = (root / "data" / tree / split / f"leaf-{tree}-{split}-{index}"
                              / "snapshots" / "v1" / "model.safetensors")
                checkpoint.parent.mkdir(parents=True)
                (checkpoint.parents[2] / "refs").mkdir()
                (checkpoint.parents[2] / "refs" / "main").write_text("v1\n")
                chosen = [CIFAR100_CLASSES[(i + index) % 100] for i in range(50)]
                save_file({key: roots[tree] + noise * torch.randn(5, 7, generator=generator)},
                          str(checkpoint), metadata={"dataset_chosen_targets": repr(chosen)})
    return roots


class TreeEstimationTests(unittest.TestCase):
    def test_purity_and_confusion(self):
        truth, assigned = [0, 0, 1, 1, 2, 2], [1, 1, 0, 0, 2, 0]
        self.assertAlmostEqual(te.purity(truth, assigned), 5 / 6)
        self.assertEqual(te.confusion(truth, assigned, 3, 3), [[0, 2, 0], [2, 0, 0], [1, 0, 1]])
        self.assertEqual(te.purity([0, 1, 2], [0, 1, 2]), 1.0)

    def test_relative_distance(self):
        reference = torch.ones(3, 4)
        self.assertAlmostEqual(te.relative_distance(reference * 1.1, reference), 10.0, places=4)
        self.assertEqual(te.relative_distance(reference, reference), 0.0)

    def test_distance_and_cluster_on_separated_trees(self):
        generator = torch.Generator().manual_seed(3)
        with tempfile.TemporaryDirectory(prefix="tree-estimation-") as directory:
            root = Path(directory)
            build(root, 53, 0.02, generator)
            out = root / "out"
            data, refs = str(root / "data"), str(root / "references")
            summary = te.run_distance(data, refs, [53], out)["53"]
            self.assertEqual(summary["nearest_root_accuracy"], 1.0)
            self.assertGreater(summary["worst_gap"], 10)
            self.assertGreater(summary["worst_leaf_margin_abs"], 0)
            self.assertEqual(set(summary["per_tree"]), set(te.TREES))
            result = te.run_cluster(data, [53], out, seeds=(0, 1), n_init=3)["53"]
            self.assertEqual(result["n_leaves"], 36)
            self.assertEqual(result["all_purity_min"], 1.0)
            self.assertEqual(result["heldout_purity_min"], 1.0)
            assignments = json.loads((out / "cluster_assignments_L53.json").read_text())
            self.assertEqual(len(assignments), 36)
            self.assertTrue({"leaf", "tree", "split", "cluster"} <= assignments[0].keys())
            self.assertTrue((out / "distance.csv").is_file() and (out / "cluster.csv").is_file())

    def test_overlapping_trees_are_not_perfectly_pure(self):
        generator = torch.Generator().manual_seed(4)
        with tempfile.TemporaryDirectory(prefix="tree-estimation-") as directory:
            root = Path(directory)
            build(root, 53, 50.0, generator)             # noise swamps the roots
            result = te.run_cluster(str(root / "data"), [53], root / "out", seeds=(0,), n_init=3)["53"]
            self.assertLess(result["all_purity_mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
