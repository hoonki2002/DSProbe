"""Reference-inclusive FLOP accounting checked against real model forwards."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import contextlib
import csv
import io
import json
import tempfile
import unittest

import torch
from torch.utils.flop_counter import FlopCounterMode

from dsprobe.costs import report_costs
from dsprobe.flops import forward_flops
from dsprobe.models import ModelConfig, build_model
from dsprobe.probe_sweep import summarize


MATRIX_OPERATORS = {"aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm"}


def measured_forward(config, batch_size):
    """Count the implemented graph independently of the analytic formula."""
    reference = torch.zeros(config.input_shape) if config.family == "gramdiff" else None
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, reference=reference).to("meta")
    weights = torch.empty(batch_size, *config.input_shape, device="meta")
    with FlopCounterMode(display=False) as counter:
        model(weights)
    operations = counter.get_flop_counts()["Global"]
    return sum(count for operation, count in operations.items()
               if str(operation) in MATRIX_OPERATORS) / batch_size


class ForwardFlopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formula_matches_actual_forward_graph(self):
        # Distinct dimensions prevent accidentally interchanging m/n or
        # silently using the publication's default projection/head sizes.
        cases = (((7, 3), 1, 1, 2, 5, 4),
                 ((4, 4), 3, 8, 7, 6, 9),
                 ((3, 8), 5, 11, 4, 9, 3))
        for family, trainable in (("mvprobe", False), ("probex", False),
                                  ("gramdiff", False), ("gramdiff", True)):
            for shape, probes, batch_size, projection, representation, classes in cases:
                with self.subTest(family=family, trainable=trainable, shape=shape,
                                  probes=probes, batch_size=batch_size):
                    config = ModelConfig(input_shape=shape, family=family,
                                         n_probes=probes, proj_dim=projection,
                                         rep_dim=representation, num_classes=classes,
                                         reference_trainable=trainable)
                    actual = measured_forward(config, batch_size)
                    analytic = forward_flops(
                        *shape, probes, branches=1 if family == "probex" else 4,
                        has_reference=family == "gramdiff", batch_size=batch_size,
                        proj_dim=projection, rep_dim=representation,
                        num_classes=classes)
                    self.assertAlmostEqual(analytic, actual, places=8)

    def test_batch_sharing_is_counted_once(self):
        # MVProbe is a different model, not an alternate reference-excluded
        # metric for the R-DSProbe model.
        baseline = forward_flops(7, 3, 5, batch_size=1)
        self.assertEqual(baseline, forward_flops(7, 3, 5, batch_size=128))
        per_batch = forward_flops(7, 3, 5, has_reference=True, batch_size=1) - baseline
        for batch_size in (2, 7, 128):
            total = forward_flops(7, 3, 5, has_reference=True, batch_size=batch_size)
            self.assertAlmostEqual((total - baseline) * batch_size, per_batch)
            self.assertGreater(total, baseline)

    def test_published_comparison_counts(self):
        for shape, probes, mvprobe, r_dsprobe in (
            ((1024, 256), 6, 553750528, 26202112),
            ((768, 768), 24, 1073844224, 202756096),
            ((768, 768), 32, 1073844224, 270307328),
            ((768, 768), 96, 1073844224, 810717184),
        ):
            with self.subTest(shape=shape, probes=probes):
                self.assertEqual(forward_flops(*shape, 128), mvprobe)
                self.assertEqual(forward_flops(*shape, probes, has_reference=True),
                                 r_dsprobe)


class CostReportTests(unittest.TestCase):
    def test_cost_report_has_only_reference_inclusive_comparison(self):
        expected = {
            "ResNet": (6, 553750528, 26202112),
            "SupViT": (24, 1073844224, 202756096),
            "MAE": (32, 1073844224, 270307328),
            "DINO": (96, 1073844224, 810717184),
        }
        fields = {"model", "mvprobe_layer", "r_dsprobe_layer", "input_rows",
                  "input_columns", "mvprobe_probes", "r_dsprobe_probes", "batch_size",
                  "mvprobe_flops_per_example", "r_dsprobe_flops_per_example", "reduction"}
        with tempfile.TemporaryDirectory(prefix="dsprobe-cost-report-") as directory:
            output = Path(directory)
            original_report = report_costs(output, batch_size=128)
            comparison = json.loads((output / "probe_efficiency.json").read_text(encoding="utf-8"))
            records = comparison["records"]
            with (output / "probe_efficiency.csv").open(newline="") as stream:
                csv_records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 4)
        self.assertEqual(len(csv_records), 4)
        for record, csv_record in zip(records, csv_records):
            with self.subTest(model=record["model"]):
                probes, baseline, reference = expected[record["model"]]
                self.assertEqual(set(record), fields)
                self.assertEqual(set(csv_record), fields)
                self.assertEqual(record["r_dsprobe_probes"], probes)
                self.assertEqual(record["mvprobe_probes"], 128)
                self.assertEqual(record["batch_size"], 128)
                self.assertEqual(record["mvprobe_flops_per_example"], baseline)
                self.assertEqual(record["r_dsprobe_flops_per_example"], reference)
                self.assertEqual(float(csv_record["r_dsprobe_flops_per_example"]), reference)
                self.assertAlmostEqual(record["reduction"], baseline / reference)
        # Existing fixed-r=128 cost estimates continue to include reference
        # products and stay distinct from the lower-r comparison above.
        fixed = {(record["model"], record["variant"]): record
                 for record in original_report["records"]}
        self.assertEqual(fixed["SupViT", "mvprobe"]["estimated_3x_forward_flops_per_example"],
                         3221532672)
        self.assertEqual(fixed["SupViT", "gramdiff"]["estimated_3x_forward_flops_per_example"],
                         3242766336)


class SweepCostTests(unittest.TestCase):
    def _result(self, *, method="r_dsprobe", family="gramdiff", seed=1,
                batch_size=7, probes=3, shape=(7, 3), layer=59):
        model = ModelConfig(input_shape=shape, n_probes=probes, proj_dim=2,
                            rep_dim=5, num_classes=4, family=family,
                            reference_trainable=method == "r_dsprobe")
        return {
            "model": "SupViT", "method": method, "layer": layer, "seed": seed,
            "parameters": 123,
            "contract": {"model": asdict(model), "training": {"batch_size": batch_size}},
            "results": {"id": {"test": {"accuracy": 75.0}}},
        }

    def _summarize(self, results, *, batch_size=None):
        with tempfile.TemporaryDirectory(prefix="dsprobe-flop-summary-") as directory:
            root = Path(directory)
            for index, result in enumerate(results):
                folder = root / str(index)
                folder.mkdir()
                (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(SimpleNamespace(output=root, flops_batch_size=batch_size))
            with (root / "probe_sweep_summary.csv").open(newline="") as stream:
                return list(csv.DictReader(stream))

    def test_summary_uses_recorded_model_and_training_dimensions(self):
        result = self._result()
        row, = self._summarize([result])
        config = ModelConfig(**result["contract"]["model"])
        self.assertAlmostEqual(float(row["forward_flops_per_example"]),
                               measured_forward(config, 7))
        self.assertEqual(int(row["batch_size"]), 7)
        self.assertEqual((int(row["proj_dim"]), int(row["rep_dim"]),
                          int(row["num_classes"])), (2, 5, 4))
        self.assertNotIn("probing_flops_per_example", row)
        self.assertFalse(any("exclud" in key for key in row))
        self.assertEqual(row["reduction_vs_mvprobe_128"], "")

    def test_summary_batch_override_and_compatible_baseline(self):
        results = [self._result(),
                   self._result(method="mvprobe", family="mvprobe", probes=128)]
        rows = {row["method"]: row for row in self._summarize(results, batch_size=128)}
        reference = rows["r_dsprobe"]
        self.assertEqual(int(reference["batch_size"]), 128)
        config = ModelConfig(**results[0]["contract"]["model"])
        reference_flops = measured_forward(config, 128)
        baseline_flops = measured_forward(ModelConfig(**results[1]["contract"]["model"]), 128)
        self.assertEqual(float(reference["forward_flops_per_example"]), reference_flops)
        self.assertEqual(float(reference["mvprobe_128_flops_per_example"]), baseline_flops)
        self.assertAlmostEqual(float(reference["reduction_vs_mvprobe_128"]),
                               baseline_flops / reference_flops)

    def test_summary_counts_frozen_reference_products(self):
        result = self._result(method="dsprobe")
        row, = self._summarize([result])
        self.assertAlmostEqual(float(row["forward_flops_per_example"]),
                               measured_forward(ModelConfig(**result["contract"]["model"]), 7))

    def test_summary_rejects_mixed_cost_contracts(self):
        for changes in ({"batch_size": 8}, {"shape": (8, 3)}, {"layer": 53}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._summarize([self._result(), self._result(seed=2, **changes)])

    def test_summary_rejects_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            self._summarize([self._result(), self._result()])


if __name__ == "__main__":
    unittest.main()
