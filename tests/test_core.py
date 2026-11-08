"""Small regression tests; all generated banks live in temporary directories."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from dsprobe.cap import ReferenceStepCap
from dsprobe.analysis import aggregate, collect_results
from dsprobe.experiments import Job, configurations, jobs_from_suite, read_suite
from dsprobe.flops import forward_flops
from dsprobe.modelj_data import CIFAR100_CLASSES, resolve_layer_name
from dsprobe.models import ModelConfig, build_model
from dsprobe.probe_sweep import summarize
from dsprobe.runner import load_modelj_inputs, run_job


ROOT = Path(__file__).resolve().parents[1]


def make_bank(root, model="SupViT", layer=59):
    generator = torch.Generator().manual_seed(17)
    key = resolve_layer_name(model, layer)
    for split in ("train", "val", "test"):
        for index in range(4):
            folder = root / model / split / f"models--example--{split}-{index}"
            target = folder / "snapshots" / "v1" / "model.safetensors"
            target.parent.mkdir(parents=True)
            (folder / "refs").mkdir()
            (folder / "refs" / "main").write_text("v1", encoding="utf-8")
            names = CIFAR100_CLASSES[index % 2::2]
            save_file({key: torch.randn(3, 2, generator=generator)}, str(target),
                      metadata={"dataset_chosen_targets": repr(list(names))})
    return key


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_layers_and_capacity(self):
        suite = read_suite(ROOT / "configs/modelj_fixed.json")
        jobs = list(jobs_from_suite(suite))
        self.assertEqual(len(jobs), 80)
        expected = {
            "probex4": (98, 59, 64, 59), "mvprobe": (55, 59, 64, 47),
            "dsprobe": (98, 53, 65, 47), "r_dsprobe": (98, 53, 65, 47),
        }
        for method, layers in expected.items():
            observed = tuple(next(j.layer for j in jobs if j.model == model and j.method == method)
                             for model in ("ResNet", "SupViT", "MAE", "DINO"))
            self.assertEqual(observed, layers)
        job = next(j for j in jobs if j.method == "probex4")
        model, _ = configurations(job, (3, 2), n_probes=7, proj_dim=5, rep_dim=11)
        self.assertEqual((model.n_probes, model.proj_dim, model.rep_dim), (28, 5, 11))

    def test_single_and_higher_layers(self):
        for name in ("single_branch", "higher_order"):
            jobs = list(jobs_from_suite(read_suite(ROOT / f"configs/{name}.json")))
            for model, layer in zip(("ResNet", "SupViT", "MAE", "DINO"), (55, 59, 64, 47)):
                self.assertEqual({j.layer for j in jobs if j.model == model}, {layer})

    def test_probex_and_all_single_response_training_defaults(self):
        suite = read_suite(ROOT / "configs/single_branch.json")
        for method in ("probex", *suite["methods"]):
            with self.subTest(method=method):
                model, training = configurations(Job("membership", "SupViT", 59, method, 1), (3, 2))
                self.assertEqual((training.epochs, training.learning_rate, training.weight_decay),
                                 (200, 2e-3, 0.0))
                if method.startswith("linear_"):
                    self.assertFalse(model.standardize)
                    self.assertFalse(training.evaluate_final_epoch)
                elif method.startswith("gram_"):
                    self.assertTrue(model.standardize)
                    self.assertTrue(training.evaluate_final_epoch)
        for method in ("probex4", "mvprobe", "dsprobe", "r_dsprobe", "poly6", "poly8",
                       "duplicate6", "duplicate8"):
            _, training = configurations(Job("membership", "SupViT", 59, method, 1), (3, 2))
            self.assertEqual((training.epochs, training.learning_rate, training.weight_decay),
                             (500, 3e-4, 1e-5))
        _, training = configurations(Job("membership", "SupViT", 59, "gram_left_learned", 1),
                                     (3, 2), epochs=3, learning_rate=1e-4, weight_decay=2e-5)
        self.assertEqual((training.epochs, training.learning_rate, training.weight_decay),
                         (3, 1e-4, 2e-5))

    def test_model_layer_overrides(self):
        suite = read_suite(ROOT / "configs/modelj_fixed.json")
        jobs = list(jobs_from_suite(suite, models=["SupViT"], model_layers={"SupViT": [11]}))
        self.assertEqual({j.layer for j in jobs}, {11})
        with self.assertRaises(ValueError):
            list(jobs_from_suite(suite, layers=[1], model_layers={"SupViT": [11]}))

    def test_sd_fair_budget(self):
        jobs = list(jobs_from_suite(read_suite(ROOT / "configs/sd_lora.json")))
        self.assertEqual(len(jobs), 30)
        training_configs = []
        for method in ("probex4", "mvprobe", "r_dsprobe"):
            job = next(j for j in jobs if j.method == method)
            model, training = configurations(job, (3, 2))
            self.assertEqual(model.n_probes, 512 if method == "probex4" else 128)
            self.assertEqual(training.epochs, 500)
            training_configs.append(training)
        self.assertEqual(training_configs[0], training_configs[1])
        self.assertEqual(training_configs[1], training_configs[2])

    def test_gram_difference_matches_its_definition(self):
        """The Gram branches must equal (XX^T - GG^T)U, not the Gram of X - G."""
        torch.manual_seed(0)
        shape, probes = (6, 5), 3
        reference = torch.randn(*shape).double()
        config = ModelConfig(input_shape=shape, family="gramdiff", n_probes=probes,
                             proj_dim=4, rep_dim=5, reference_trainable=True)
        model = build_model(config, reference=reference).double()
        x = torch.randn(4, *shape).double()
        got = model.raw_probe_responses(x)
        g, gt = model.x_star, model.x_star.transpose(0, 1)
        xt = x.transpose(1, 2)
        expected = {
            "xu": (x - g) @ model.probes_xu.weight,
            "xtu": (x - g).transpose(1, 2) @ model.probes_xtu.weight,
            "xxtu": (x @ xt - g @ gt) @ model.probes_xxtu.weight,
            "xtxu": (xt @ x - gt @ g) @ model.probes_xtxu.weight,
        }
        for name, value in expected.items():
            torch.testing.assert_close(got[name], value)

    def test_probe_sweep_suite_and_cost(self):
        """Every probe count reaches its own directory, and cost scales with r."""
        suite = read_suite(Path("configs/probe_sweep.json"))
        jobs = list(jobs_from_suite(suite, models=["SupViT"], seeds=[1]))
        self.assertEqual({(job.method, job.layer) for job in jobs},
                         {("mvprobe", 59), ("r_dsprobe", 53)})
        for job in jobs:
            model, _ = configurations(job, (3, 2), n_probes=7)
            self.assertEqual(model.n_probes, 7)
        keys = {f"probes_{probes}/{job.key}" for probes in (4, 8) for job in jobs}
        self.assertEqual(len(keys), 2 * len(jobs))
        head = 2 * 512 * 100  # the classifier does not depend on the probe count
        self.assertEqual(forward_flops(768, 768, 8, has_reference=True) - head,
                         2 * (forward_flops(768, 768, 4, has_reference=True) - head))

    def test_frozen_reference_and_higher_controls(self):
        config = ModelConfig(input_shape=(3, 2), n_probes=2, proj_dim=3, rep_dim=4,
                             family="gramdiff", reference_trainable=False)
        model = build_model(config, reference=torch.ones(3, 2))
        self.assertNotIn("x_star", dict(model.named_parameters()))
        x = torch.randn(4, 3, 2)
        for poly, control, branches in (("poly6", "duplicate6", 6), ("poly8", "duplicate8", 8)):
            a = build_model(replace(config, family=poly))
            b = build_model(replace(config, family=control))
            self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
            self.assertEqual(len(a.raw_probe_responses(x)), branches)
            self.assertEqual(a(x).shape, (4, 100))

    def test_cap_is_per_step(self):
        parameter = torch.nn.Parameter(torch.ones(3, 2))
        initial = parameter.detach().clone()
        cap = ReferenceStepCap(parameter, initial, 1e-3)
        for _ in range(3):
            before = cap.capture()
            with torch.no_grad():
                parameter.add_(1)
            stats = cap.apply(before)
            self.assertTrue(stats["clipped"])
            self.assertLessEqual(stats["applied_relative_update"], 1.001e-3)
        self.assertGreater(float(torch.linalg.vector_norm(parameter - initial) / initial.norm()), 2e-3)

    def test_reference_distance_reads_saved_states(self):
        """Table 15's aggregation must use the saved x_star, not re-derive it."""
        from dsprobe.distance import relative_distance, report_distances
        pretrained = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        self.assertAlmostEqual(relative_distance(pretrained, pretrained), 0.0)
        self.assertAlmostEqual(relative_distance(pretrained * 2, pretrained), 100.0)
        with self.assertRaises(ValueError):
            relative_distance(torch.zeros(3, 2), pretrained)
        with tempfile.TemporaryDirectory(prefix="dsprobe-distance-") as directory:
            root = Path(directory)
            data, refs, output = root / "data", root / "refs", root / "results"
            make_bank(data)
            reference = refs / "SupViT" / "layer_59.safetensors"
            reference.parent.mkdir(parents=True)
            save_file({"reference": torch.ones(3, 2)}, str(reference))
            job = next(job for job in jobs_from_suite(
                read_suite(ROOT / "configs/modelj_fixed.json"),
                models=["SupViT"], methods=["r_dsprobe"], layers=[59], seeds=[1]))
            run_job(job, data_root=data, reference_root=refs, text_root=None, output=output,
                    device="cpu", n_probes=2, proj_dim=3, rep_dim=5, epochs=1,
                    batch_size=2, num_workers=0, save_weights=True)
            rows = report_distances(output, data, refs, root / "summary")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seeds"], 1)
            self.assertTrue((root / "summary" / "reference_distance.csv").is_file())
        # Without saved states the command must refuse rather than guess.
        with tempfile.TemporaryDirectory(prefix="dsprobe-distance-") as directory:
            root = Path(directory)
            data, refs, output = root / "data", root / "refs", root / "results"
            make_bank(data)
            reference = refs / "SupViT" / "layer_59.safetensors"
            reference.parent.mkdir(parents=True)
            save_file({"reference": torch.ones(3, 2)}, str(reference))
            job = next(job for job in jobs_from_suite(
                read_suite(ROOT / "configs/modelj_fixed.json"),
                models=["SupViT"], methods=["r_dsprobe"], layers=[59], seeds=[1]))
            run_job(job, data_root=data, reference_root=refs, text_root=None, output=output,
                    device="cpu", n_probes=2, proj_dim=3, rep_dim=5, epochs=1,
                    batch_size=2, num_workers=0)
            with self.assertRaises(FileNotFoundError):
                report_distances(output, data, refs, root / "summary")

    def test_statnn_is_not_a_single_layer_neural_model(self):
        """StatNN has a full-model XGBoost runner, never the legacy seven-input MLP."""
        with self.assertRaises(ValueError):
            ModelConfig(input_shape=(4, 3), family="statnn", rep_dim=8, num_classes=5)

    def test_all_membership_methods_one_epoch_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="dsprobe-test-") as directory:
            root = Path(directory)
            data, refs, output = root / "data", root / "refs", root / "results"
            make_bank(data)
            reference = refs / "SupViT" / "layer_59.safetensors"
            reference.parent.mkdir(parents=True)
            save_file({"reference": torch.ones(3, 2)}, str(reference))
            jobs = {}
            for suite in ("modelj_fixed", "higher_order", "single_branch", "modelj_layerwise"):
                for job in jobs_from_suite(read_suite(ROOT / f"configs/{suite}.json"),
                                           models=["SupViT"], layers=[59], seeds=[1]):
                    jobs[job.method] = job
            kwargs = dict(data_root=data, reference_root=refs, text_root=None, output=output,
                          device="cpu", n_probes=2, proj_dim=3, rep_dim=5, epochs=1,
                          batch_size=2, num_workers=0)
            for job in jobs.values():
                result = run_job(job, **kwargs)
                self.assertTrue(0 <= result["results"]["id"]["test"]["accuracy"] <= 100)
            job = jobs["dsprobe"]
            run_job(job, resume=True, **kwargs)
            with self.assertRaises(FileExistsError):
                run_job(job, **kwargs)
            with self.assertRaises(ValueError):
                run_job(job, resume=True, **{**kwargs, "n_probes": 3})
            self.assertEqual(len(list(output.rglob("result.json"))), len(jobs))
            self.assertFalse(list(output.rglob("*.safetensors")))
            for path in output.rglob("result.json"):
                self.assertNotIn(str(root), path.read_text(encoding="utf-8"))
            from dsprobe.analysis import collect_results, aggregate
            self.assertEqual(len(aggregate(collect_results(output))), len(jobs))


class ResultRegressionTests(unittest.TestCase):
    """Exercise real result files across training, serialization and aggregation."""

    CASES = (("dsprobe_rand", "mean"), ("dsprobe_rand_norm", "mean"),
             ("r_dsprobe", "random"), ("r_dsprobe", "random_norm"))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="dsprobe-result-test-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        data, refs = cls.root / "data", cls.root / "refs"
        make_bank(data)
        reference = refs / "SupViT" / "layer_59.safetensors"
        reference.parent.mkdir(parents=True)
        save_file({"reference": torch.ones(3, 2)}, str(reference))
        prepared = load_modelj_inputs(Job("membership", "SupViT", 59, "mvprobe", 1), data)
        cls.kwargs = dict(data_root=data, reference_root=refs, text_root=None,
                          device="cpu", n_probes=2, proj_dim=3, rep_dim=5,
                          epochs=1, batch_size=2, num_workers=0, prepared=prepared)
        cls.results, cls.outputs = {}, {}
        with contextlib.redirect_stdout(io.StringIO()):
            for method, initialization in cls.CASES:
                output = cls.root / f"{method}_{initialization}"
                cls.outputs[method, initialization] = output
                cls.results[method, initialization] = [
                    run_job(Job("membership", "SupViT", 59, method, seed),
                            output=output, learned_init=initialization, **cls.kwargs)
                    for seed in (1, 2)
                ]
            cls.sweep_output = cls.root / "sweep"
            cls.sweep_result = run_job(
                Job("membership", "SupViT", 59, "mvprobe", 1),
                output=cls.sweep_output / "probes_2", **cls.kwargs)

    def _collect_copies(self, results):
        """Alter only temporary result fixtures, never the original run artifacts."""
        with tempfile.TemporaryDirectory(dir=self.root, prefix="fixtures-") as directory:
            for index, result in enumerate(results):
                folder = Path(directory) / str(index)
                folder.mkdir()
                (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return collect_results(directory)

    def test_probe_sweep_summarizes_real_result(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summarize(SimpleNamespace(output=self.sweep_output))
        with (self.sweep_output / "probe_sweep_summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["model"], row["method"], int(row["layer"])),
                         ("SupViT", "mvprobe", 59))
        self.assertEqual((int(row["probes"]), int(row["seeds"])), (2, 1))
        self.assertNotIn("probing_flops_per_example", row)
        self.assertEqual(float(row["forward_flops_per_example"]),
                         forward_flops(3, 2, 2, proj_dim=3, rep_dim=5,
                                       num_classes=100, batch_size=2))
        self.assertEqual(int(row["parameters"]), self.sweep_result["parameters"])
        self.assertAlmostEqual(float(row["mean_test_accuracy"]),
                               self.sweep_result["results"]["id"]["test"]["accuracy"], places=3)

    def test_random_reference_variants_aggregate_across_seeds(self):
        for case in self.CASES:
            with self.subTest(case=case):
                results = self.results[case]
                hashes = [result["contract"]["initial_reference_sha256"] for result in results]
                self.assertNotEqual(*hashes)
                if "norm" in case[0] or case[1] == "random_norm":
                    targets = [result["contract"]["random_norm_target_sha256"] for result in results]
                    self.assertEqual(targets[0], targets[1])
                    self.assertEqual(len(targets[0]), 64)
                summary = aggregate(collect_results(self.outputs[case]))
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]["n_seeds"], 2)
                self.assertFalse(summary[0]["complete_five_seeds"])
                accuracies = [result["results"]["id"]["test"]["accuracy"] for result in results]
                self.assertAlmostEqual(summary[0]["test_mean"], sum(accuracies) / 2)
                self.assertAlmostEqual(summary[0]["test_sd"], abs(accuracies[0] - accuracies[1]) / 2)

    def test_aggregation_keeps_scientific_configuration_guards(self):
        for field, value in (("learning_rate", 9e-4), ("cap", 1e-2)):
            with self.subTest(training=field):
                results = deepcopy(self.results["r_dsprobe", "random"])
                results[1]["contract"]["training"][field] = value
                with self.assertRaisesRegex(ValueError, "different scientific configurations"):
                    aggregate(self._collect_copies(results))

        # Both initializations omit seed-specific hashes. Their shared target
        # provenance must not make different initialization policies equivalent.
        results = deepcopy(self.results["r_dsprobe", "random_norm"])
        results[1]["contract"]["learned_reference_init"] = "random"
        with self.subTest(reference="initialization"), self.assertRaisesRegex(
                ValueError, "different scientific configurations"):
            aggregate(self._collect_copies(results))

        results = deepcopy(self.results["dsprobe_rand_norm", "mean"])
        results[1]["contract"]["random_norm_target_sha256"] = "0" * 64
        with self.subTest(reference="norm_target"), self.assertRaisesRegex(
                ValueError, "different scientific configurations"):
            aggregate(self._collect_copies(results))

        # The frozen random and pretrained methods have the same architecture;
        # changing this fixture's method makes differing reference hashes invalid.
        results = deepcopy(self.results["dsprobe_rand", "mean"])
        for result in results:
            result["method"] = result["contract"]["job"]["method"] = "dsprobe"
        with self.subTest(reference="nonrandom_hash"), self.assertRaisesRegex(
                ValueError, "different scientific configurations"):
            aggregate(self._collect_copies(results))

    def test_legacy_random_norm_keeps_reference_hash_guard(self):
        for case in (("dsprobe_rand_norm", "mean"), ("r_dsprobe", "random_norm")):
            with self.subTest(case=case):
                results = deepcopy(self.results[case])
                for result in results:
                    del result["contract"]["random_norm_target_sha256"]
                with self.assertRaisesRegex(ValueError, "different scientific configurations"):
                    aggregate(self._collect_copies(results))

    def test_gpu_random_reference_hash_matches_cpu(self):
        device = ("cuda:0" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else None)
        if device is None:
            self.skipTest("No CUDA or MPS device available")
        for method, initialization in self.CASES:
            with self.subTest(method=method, initialization=initialization, device=device):
                cpu = self.results[method, initialization][0]
                # Run real loading, reference construction and fingerprinting on
                # the GPU. Training is irrelevant to the pre-training hash bug.
                with patch("dsprobe.runner.fit", return_value=(deepcopy(cpu), {})) as fit:
                    result = run_job(
                        Job("membership", "SupViT", 59, method, 1),
                        output=self.root / f"gpu_{method}_{initialization}",
                        learned_init=initialization, **{**self.kwargs, "device": device})
                fit.assert_called_once()
                self.assertEqual(fit.call_args.args[0].x_star.device.type, torch.device(device).type)
                self.assertEqual(result["contract"]["device_type"], torch.device(device).type)
                self.assertEqual(result["contract"]["initial_reference_sha256"],
                                 cpu["contract"]["initial_reference_sha256"])
                self.assertEqual(result["contract"].get("random_norm_target_sha256"),
                                 cpu["contract"].get("random_norm_target_sha256"))


if __name__ == "__main__":
    unittest.main()
