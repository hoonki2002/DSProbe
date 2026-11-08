"""Portable CPU checks for the tree-label-noise banks."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from dsprobe import contaminate
from dsprobe.modelj_data import ModelJDataset
from dsprobe.tree_estimation import TREES
from tests.test_tree_estimation import build


class ContaminateTests(unittest.TestCase):
    def test_nearest_foreign_leaves_replace_a_fraction(self):
        generator = torch.Generator().manual_seed(5)
        layers = {tree: 53 for tree in TREES}
        with tempfile.TemporaryDirectory(prefix="contaminate-") as directory:
            root = Path(directory)
            build(root, 53, 0.02, generator, sizes=(("train", 10), ("val", 3), ("test", 3)))
            out = root / "contam"
            planned = contaminate.build(root / "data", out, [0, 20, 50], seed=1, layers=layers)
            for tree in TREES:
                fractions = planned[tree]["fractions"]
                self.assertEqual(fractions["0"]["k"], 0)
                self.assertEqual((fractions["20"]["k"], fractions["50"]["k"]), (2, 5))
                # nested: the 20% foreign set is the first part of the 50% one
                self.assertEqual(fractions["50"]["added"][:2], fractions["20"]["added"])
                self.assertEqual(fractions["50"]["removed"][:2], fractions["20"]["removed"])
                for p in (0, 20, 50):
                    bank = out / f"p{p}"
                    train = ModelJDataset(bank, tree, 53, "train")
                    self.assertEqual(len(train), 10)                # size is preserved
                    self.assertEqual(len(ModelJDataset(bank, tree, 53, "val")), 3)
                    self.assertEqual(len(ModelJDataset(bank, tree, 53, "test")), 3)
                names = {p.parents[2].name for p in ModelJDataset(out / "p50", tree, 53, "train").paths}
                foreign = [n for n in names if tree not in n]
                self.assertEqual(len(foreign), 5)
                self.assertTrue((out / "manifest.json").is_file())

    def test_too_large_fraction_is_an_error(self):
        generator = torch.Generator().manual_seed(6)
        layers = {tree: 53 for tree in TREES}
        with tempfile.TemporaryDirectory(prefix="contaminate-") as directory:
            root = Path(directory)
            build(root, 53, 0.02, generator, sizes=(("train", 4), ("val", 1), ("test", 1)))
            with self.assertRaises(ValueError):
                contaminate.build(root / "data", root / "c", [300], layers=layers)


if __name__ == "__main__":
    unittest.main()
