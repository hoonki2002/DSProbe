"""CPU SD-LoRA integration checks using temporary banks and synthetic text features.

These fixtures test execution and data contracts, not predictive performance.
No CLIP checkpoint, remote dataset, GPU, or persistent model cache is needed.
"""

from pathlib import Path
import json
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from dsprobe.experiments import Job
from dsprobe.runner import load_sd_inputs, run_job
from dsprobe.sd_data import discover_sd_classes, load_sd_dataset, load_sd_reference


DOWN = "unet.block.attn.to_q.lora.down.weight"
UP = "unet.block.attn.to_q.lora.up.weight"
SEEN = ["n00000001", "n00000002"]
HELDOUT = ["n00000003", "n00000004"]
SPLITS = ("train", "val", "test", "val_holdout", "test_holdout")


def make_sd_bank(data_root, text_root):
    """Write rank-two factors whose reconstructed layer is a 2-by-3 matrix."""
    generator = torch.Generator().manual_seed(53)
    paths = {}
    for split in SPLITS:
        paths[split] = []
        classes = HELDOUT if split.endswith("holdout") else SEEN
        for index in range(4):
            folder = data_root / "SD_200" / split / f"model_{index}"
            folder.mkdir(parents=True)
            path = folder / "pytorch_lora_weights.safetensors"
            save_file({DOWN: torch.randn(2, 3, generator=generator),
                       UP: torch.randn(2, 2, generator=generator)}, str(path),
                      metadata={"imagenet_class_id": classes[index % 2]})
            paths[split].append(path)
    text_folder = text_root / "SD_200"
    text_folder.mkdir(parents=True)
    (text_folder / "classes.json").write_text(
        json.dumps({"train": SEEN, "holdout": HELDOUT}), encoding="utf-8")
    save_file({"train": torch.randn(2, 768, generator=generator),
               "holdout": torch.randn(2, 768, generator=generator)},
              str(text_folder / "text_embeddings.safetensors"))
    return paths


class SDLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_methods_train_id_zero_shot_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            root = Path(temporary)
            data, texts, output = root / "data", root / "text", root / "results"
            make_sd_bank(data, texts)
            self.assertEqual(discover_sd_classes(data, "SD_200"), (SEEN, HELDOUT))
            kwargs = dict(data_root=data, reference_root=None, text_root=texts,
                          output=output, device="cpu", n_probes=2, proj_dim=3,
                          rep_dim=5, epochs=1, batch_size=2, eval_every=1, num_workers=0)
            for method in ("probex4", "mvprobe", "r_dsprobe"):
                with self.subTest(method=method):
                    job = Job("sd_lora", "SD_200", 0, method, 1)
                    result = run_job(job, **kwargs)
                    self.assertEqual(set(result["results"]), {"id", "zero_shot"})
                    self.assertEqual(result["contract"]["layer_name"], DOWN)
                    self.assertEqual(result["contract"]["model"]["n_probes"],
                                     8 if method == "probex4" else 2)
                    for group in ("id", "zero_shot"):
                        self.assertEqual(result["results"][group]["validation"]["epoch"], 1)
                        self.assertTrue(0 <= result["results"][group]["test"]["accuracy"] <= 100)
                    resumed = run_job(job, resume=True, **kwargs)
                    self.assertEqual(result["contract_sha256"], resumed["contract_sha256"])
                    with self.assertRaises(FileExistsError):
                        run_job(job, **kwargs)
                    with self.assertRaises(ValueError):
                        run_job(job, resume=True, **{**kwargs, "n_probes": 3})
                    with self.assertRaises(FileNotFoundError):
                        run_job(job, resume=True, save_weights=True, **kwargs)
            self.assertEqual(len(list(output.rglob("result.json"))), 3)
            self.assertFalse(list(output.rglob("*.safetensors")))
            for result_file in output.rglob("result.json"):
                self.assertNotIn(str(root), result_file.read_text(encoding="utf-8"))

    def test_split_source_collision_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            root = Path(temporary)
            data, texts = root / "data", root / "text"
            paths = make_sd_bank(data, texts)
            destination = paths["test"][0]
            destination.unlink()
            destination.symlink_to(paths["train"][0])
            with self.assertRaisesRegex(ValueError, "source overlap"):
                load_sd_inputs(Job("sd_lora", "SD_200", 0, "mvprobe", 1), data, texts)

    def test_seen_heldout_class_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            root = Path(temporary)
            paths = make_sd_bank(root / "data", root / "text")
            save_file({DOWN: torch.ones(2, 3), UP: torch.eye(2)},
                      str(paths["test_holdout"][0]),
                      metadata={"imagenet_class_id": SEEN[0]})
            with self.assertRaisesRegex(ValueError, "class IDs overlap"):
                discover_sd_classes(root / "data", "SD_200")

    def test_text_class_order_is_checked(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            root = Path(temporary)
            data, texts = root / "data", root / "text"
            make_sd_bank(data, texts)
            (texts / "SD_200" / "classes.json").write_text(
                json.dumps({"train": list(reversed(SEEN)), "holdout": HELDOUT}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class order"):
                load_sd_inputs(Job("sd_lora", "SD_200", 0, "mvprobe", 1), data, texts)

    def test_factor_precision_and_cross_split_shapes_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            root = Path(temporary)
            data, texts = root / "data", root / "text"
            paths = make_sd_bank(data, texts)
            save_file({DOWN: torch.ones(2, 3, dtype=torch.float64), UP: torch.eye(2)},
                      str(paths["train"][0]), metadata={"imagenet_class_id": SEEN[0]})
            dataset = load_sd_dataset(data, "SD_200", 0, "train")
            with self.assertRaisesRegex(ValueError, "float32 factors"):
                dataset.factors(0)
            save_file({DOWN: torch.ones(2, 3), UP: torch.eye(2)},
                      str(paths["train"][0]), metadata={"imagenet_class_id": SEEN[0]})
            for index, path in enumerate(paths["val"]):
                save_file({DOWN: torch.ones(2, 3), UP: torch.ones(4, 2)}, str(path),
                          metadata={"imagenet_class_id": SEEN[index % 2]})
            with self.assertRaisesRegex(ValueError, "shape differs across splits"):
                load_sd_inputs(Job("sd_lora", "SD_200", 0, "mvprobe", 1), data, texts)

    def test_explicit_reference_coordinates_and_layer_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-sd-test-") as temporary:
            path = Path(temporary) / "reference.safetensors"
            reference = torch.ones(2, 3)
            metadata = {"representation": "lora_update", "layer_name": DOWN}
            save_file({"reference": reference}, str(path), metadata=metadata)
            torch.testing.assert_close(load_sd_reference(path, DOWN, (2, 3)), reference)
            with self.assertRaisesRegex(ValueError, "layer_name"):
                load_sd_reference(path, "different.lora.down.weight", (2, 3))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_sd_reference(path, DOWN, (3, 2))
            save_file({"reference": reference}, str(path),
                      metadata={**metadata, "representation": "full_unet_weight"})
            with self.assertRaisesRegex(ValueError, "representation=lora_update"):
                load_sd_reference(path, DOWN, (2, 3))


if __name__ == "__main__":
    unittest.main()
