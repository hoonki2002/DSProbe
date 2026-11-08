"""Portable CPU checks for shared encoders and free response offsets."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from dsprobe import offset, shared
from dsprobe.modelj_data import CIFAR100_CLASSES, resolve_layer_name
from dsprobe.common import DEFAULT_LAYERS


class SharedOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_offset_and_reference_equal_at_initialization(self):
        generator = torch.Generator().manual_seed(40)
        reference = torch.randn(4, 6, generator=generator)
        x = torch.randn(3, 4, 6, generator=generator)
        torch.manual_seed(7)
        free = offset.OffsetProbe((4, 6), 3, "mean_response", reference, 2, 5)
        torch.manual_seed(7)
        tied = offset.OffsetProbe((4, 6), 3, "learned_reference", reference, 2, 5)
        torch.testing.assert_close(free(x), tied(x), rtol=0, atol=0)
        free(x).sum().backward()
        tied(x).sum().backward()
        self.assertIsNotNone(free.B_xu.grad)
        self.assertIsNotNone(tied.front.W_star.grad)
        self.assertIsNone(free.front.W_star)
        self.assertFalse(hasattr(tied, "B_xu"))

    def test_dsprobe_reference_is_frozen_and_matches_learned_at_initialization(self):
        generator = torch.Generator().manual_seed(41)
        root = torch.randn(4, 6, generator=generator)
        x = torch.randn(3, 4, 6, generator=generator)
        torch.manual_seed(9)
        learned = shared.SpecialistProbe((4, 6), 3, root, 2, 5, trainable_reference=True)
        torch.manual_seed(9)
        fixed = shared.SpecialistProbe((4, 6), 3, root, 2, 5, trainable_reference=False)
        torch.testing.assert_close(learned(x), fixed(x), rtol=0, atol=0)
        self.assertIn("front.W_star", dict(learned.named_parameters()))
        self.assertNotIn("front.W_star", dict(fixed.named_parameters()))
        self.assertIn("front.W_star", dict(fixed.named_buffers()))
        fixed(x).sum().backward()
        self.assertIsNone(fixed.front.W_star.grad)
        before = fixed.front.W_star.clone()
        torch.optim.Adam(fixed.parameters(), lr=1.0).step()
        torch.testing.assert_close(fixed.front.W_star, before, rtol=0, atol=0)

    def test_shared_parameter_partition(self):
        shapes = {"SupViT": (4, 6), "MAE": (5, 4), "DINO": (6, 5)}
        refs = {name: torch.ones(shape) for name, shape in shapes.items()}
        model = shared.SharedProbe(shapes, 3, refs, 2, 5)
        report = model.parameter_report()
        self.assertEqual(report["shared_encoder_head"], (4 * 2 * 3 + 1) * 5 + (5 + 1) * 100)
        self.assertEqual(report["total"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(report["total"], report["shared_encoder_head"] +
                         sum(report["per_tree_frontend"].values()))
        for name, shape in shapes.items():
            self.assertEqual(model(name, torch.randn(2, *shape)).shape, (2, 100))

    def test_all_variants_one_epoch(self):
        with tempfile.TemporaryDirectory(prefix="shared-offset-") as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(12)
            for name, layer in DEFAULT_LAYERS.items():
                key = resolve_layer_name(name, layer)
                for split, count in (("train", 4), ("val", 2), ("test", 2)):
                    for index in range(count):
                        checkpoint = root / "data" / name / split / f"leaf-{split}-{index}" / "snapshots" / "v1" / "model.safetensors"
                        checkpoint.parent.mkdir(parents=True)
                        chosen = [CIFAR100_CLASSES[(i + index) % 100] for i in range(50)]
                        save_file({key: torch.randn(4, 6, generator=generator)}, str(checkpoint),
                                  metadata={"dataset_chosen_targets": repr(chosen)})
                (root / "references").mkdir(exist_ok=True)
                save_file({key: torch.randn(4, 6, generator=generator)},
                          str(root / "references" / f"{name}.safetensors"))
            common = ["--data-root", str(root / "data"), "--reference-root", str(root / "references"),
                      "--seeds", "1", "--epochs", "1",
                      "--eval-every", "1", "--n-probes", "3", "--proj-dim", "2", "--rep-dim", "5",
                      "--batch-size", "2", "--per-tree-batch", "2", "--threads", "1", "--device", "cpu"]
            for module in (shared, offset):
                output = root / module.__name__.split(".")[-1]
                command = common + ["--output", str(output)]
                with redirect_stdout(StringIO()):
                    module.main(command)
                    module.main(command + ["--resume"])
                    module.main(command + ["--summarize"])
                files = list(output.glob("*/seed_1.json"))
                self.assertEqual(len(files), len(module.VARIANTS))
                results = {}
                for path in files:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    self.assertTrue(result["complete"])
                    self.assertEqual(set(result["test_accuracy"]), set(DEFAULT_LAYERS))
                    self.assertTrue(all(0 <= value <= 100 for value in result["test_accuracy"].values()))
                    results[result["contract"]["protocol"]["variant"]] = result
                if module is shared:
                    # The raw per-tree row must differ from the reference row only
                    # by the reference: same runner, same schedule, one W* fewer.
                    raw, learned = results["specialist_raw"], results["specialist"]
                    self.assertEqual(raw["contract"]["protocol"]["reference_initialization"], "none")
                    self.assertEqual(learned["contract"]["protocol"]["reference_initialization"],
                                     "float32_training_mean")
                    self.assertNotIn("batch_schedule", raw["contract"]["protocol"])
                    self.assertEqual(learned["parameters"] - raw["parameters"],
                                     sum(4 * 6 for _ in DEFAULT_LAYERS))
                    # DSProbe subtracts a frozen root: same architecture as the raw row,
                    # one untrained W_pre per tree instead of a trained W*.
                    fixed = results["specialist_dsprobe"]
                    self.assertEqual(fixed["contract"]["protocol"]["reference_initialization"],
                                     "pretrained_fixed")
                    self.assertEqual(fixed["parameters"], raw["parameters"])
                    self.assertEqual(set(fixed["contract"]["pretrained_roots_sha256"]), set(DEFAULT_LAYERS))
                    self.assertEqual(results["shared_dsprobe"]["parameter_report"]["total"],
                                     results["shared_raw"]["parameter_report"]["total"])
                    self.assertEqual(results["shared_ref"]["parameter_report"]["total"] -
                                     results["shared_dsprobe"]["parameter_report"]["total"],
                                     sum(4 * 6 for _ in DEFAULT_LAYERS))
                self.assertTrue((output / "summary.csv").is_file())
            self.assertEqual(list(root.rglob("*.pt")), [])
            self.assertEqual(list(root.rglob("*.pth")), [])


if __name__ == "__main__":
    unittest.main()
