"""Offline checks for the slim layer downloader."""
from __future__ import annotations

import unittest

import torch
from safetensors.torch import save

from dsprobe import download_layers as dl


class ReadLayersTests(unittest.TestCase):
    def blob(self, big=False):
        tensors = {f"t{i}": torch.randn(6, 8) for i in range(5)}
        tensors["wide"] = torch.randn(4, 4).half()
        if big:                                   # header longer than the first read
            tensors.update({f"filler.{'x' * 120}.{i}": torch.zeros(1) for i in range(800)})
        return tensors, save(tensors, metadata={"dataset_chosen_targets": "['a']"})

    def fetcher(self, data, log):
        def fetch(start, end):
            log.append((start, end))
            return data[start:end + 1]
        return fetch

    def test_reads_only_requested_tensors(self):
        tensors, data = self.blob()
        log = []
        metadata, got = dl.read_layers(self.fetcher(data, log), ["t3", "wide"])
        self.assertEqual(metadata, {"dataset_chosen_targets": "['a']"})
        torch.testing.assert_close(got["t3"], tensors["t3"], rtol=0, atol=0)
        torch.testing.assert_close(got["wide"], tensors["wide"], rtol=0, atol=0)
        self.assertEqual(got["wide"].dtype, torch.float16)
        self.assertEqual(len(log), 3)             # header read + two tensors
        self.assertEqual(sum(e - s + 1 for s, e in log[1:]), 6 * 8 * 4 + 4 * 4 * 2)

    def test_header_larger_than_first_read(self):
        tensors, data = self.blob(big=True)
        log = []
        _, got = dl.read_layers(self.fetcher(data, log), ["t0"])
        torch.testing.assert_close(got["t0"], tensors["t0"], rtol=0, atol=0)
        self.assertEqual(len(log), 3)             # first read, rest of header, tensor

    def test_missing_tensor_is_an_error(self):
        _, data = self.blob()
        with self.assertRaises(KeyError):
            dl.read_layers(self.fetcher(data, []), ["nope"])


class RetryTests(unittest.TestCase):
    def test_rate_limit_waits_and_retries(self):
        calls, sleeps = [], []
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("429 Too Many Requests: Retry after 7 seconds")
            return "ok"
        original = dl.time.sleep
        dl.time.sleep = sleeps.append
        try:
            self.assertEqual(dl._call(flaky), "ok")
        finally:
            dl.time.sleep = original
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [9, 9])                    # Retry-After + 2

    def test_other_errors_fail_after_a_few_retries(self):
        sleeps = []
        original = dl.time.sleep
        dl.time.sleep = sleeps.append
        try:
            with self.assertRaises(ValueError):
                dl._call(lambda: (_ for _ in ()).throw(ValueError("bad")))
        finally:
            dl.time.sleep = original
        self.assertEqual(sleeps, [1, 2, 4])


class PlanNamesTests(unittest.TestCase):
    def test_default_and_per_subset_layers(self):
        default = dl.plan_names(["SupViT", "MAE"], [53, 65])
        self.assertEqual(default["SupViT"], default["MAE"])
        self.assertEqual(len(default["SupViT"]), 2)
        mixed = dl.plan_names(["ResNet", "SupViT"], [53], ["ResNet=55", "SupViT=59,64"])
        self.assertEqual(mixed["ResNet"], ["resnet.encoder.stages.2.layers.17.layer.2.convolution.weight"])
        self.assertEqual(mixed["SupViT"], ["vit.encoder.layer.7.attention.output.dense.weight",
                                           "vit.encoder.layer.8.attention.attention.value.weight"])

    def test_bad_override_is_an_error(self):
        with self.assertRaises(ValueError):
            dl.plan_names(["SupViT"], [53], ["MAE=64"])
        with self.assertRaises(ValueError):
            dl.plan_names(["SupViT"], [53], ["SupViT"])


class PresetTests(unittest.TestCase):
    def run_dry(self, *argv):
        import contextlib, io, json
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dl.main(["--dry-run", *argv])
        return json.loads(out.getvalue())

    def test_main_preset_covers_every_probed_layer(self):
        plan = self.run_dry("--preset", "main")
        self.assertEqual(set(plan["subsets"]), {"ResNet", "SupViT", "MAE", "DINO"})
        for model, layers in dl.MAIN_LAYERS.items():
            self.assertEqual(plan["tensors"][model], dl.plan_names([model], layers)[model])

    def test_preset_respects_subset_and_rejects_model_layers(self):
        plan = self.run_dry("--preset", "main", "--subset", "MAE")
        self.assertEqual(list(plan["tensors"]), ["MAE"])
        with self.assertRaises(SystemExit):
            self.run_dry("--preset", "main", "--model-layers", "MAE=64")


if __name__ == "__main__":
    unittest.main()
